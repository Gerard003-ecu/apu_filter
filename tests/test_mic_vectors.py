"""
tests/test_mic_vectors.py

Suite de pruebas exhaustiva para app/adapters/mic_vectors.py

Organización por estratos:
─────────────────────────
1. Contratos y Estructuras de Datos
2. Helpers Internos
3. Guardas Topológicas
4. Funciones de Topología Algebraica
5. Vector Físico 1: Estabilización de Flujo
6. Vector Físico 2: Parsing Topológico
7. Vector Táctico: Estructuración Lógica
8. VectorFactory
9. Composición de Vectores
10. Propiedades Algebraicas Transversales
"""

import os
import time
from dataclasses import FrozenInstanceError
from typing import Any, Dict, List
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from app.adapters.mic_vectors import (
    VectorFactory,
    VectorMetrics,
    VectorResultStatus,
    _build_error,
    _build_result,
    _elapsed_ms,
    _measure_memory_mb,
    calculate_algebraic_integrity,
    calculate_betti_numbers,
    calculate_dimensionality,
    calculate_topological_coherence,
    compose_vectors,
    validate_dimensional_isomorphism,
    validate_homological_constraints,
    validate_topological_preconditions,
    vector_parse_raw_structure,
    vector_stabilize_flux,
    vector_structure_logic,
)
from app.schemas import Stratum


# =============================================================================
# FIXTURES COMPARTIDAS
# =============================================================================
@pytest.fixture
def tmp_file(tmp_path):
    """Crea un archivo temporal válido para pruebas de existencia."""
    f = tmp_path / "test_data.csv"
    f.write_text("col1,col2\n1,2\n3,4\n")
    return str(f)


@pytest.fixture
def nonexistent_file(tmp_path):
    """Ruta a un archivo que no existe."""
    return str(tmp_path / "ghost.csv")


@pytest.fixture
def valid_physics_config():
    """Configuración mínima válida para vector de estabilización."""
    return {
        "system_capacitance": 1.5,
        "system_inductance": 0.8,
        "base_resistance": 100.0,
        "resonance_threshold": 0.85,
        "file_profile": {"type": "csv", "delimiter": ","},
    }


@pytest.fixture
def valid_profile():
    """Perfil válido para parsing."""
    return {"type": "csv", "delimiter": ",", "encoding": "utf-8"}


@pytest.fixture
def valid_topological_constraints():
    """Restricciones homológicas válidas."""
    return {
        "max_dimension": 2,
        "allow_holes": False,
        "connectivity": 0.95,
    }


@pytest.fixture
def sample_raw_records():
    """Registros crudos de ejemplo con variedad de tipos."""
    return [
        {"record_type": "material", "code": "M01", "qty": 10, "price": 5.0},
        {"record_type": "material", "code": "M02", "qty": 20, "price": 8.0},
        {"record_type": "labor", "code": "L01", "hours": 8, "rate": 15.0},
        {"record_type": "equipment", "code": "E01", "hours": 4, "rate": 50.0},
        {"record_type": "material", "code": "M03", "qty": 5, "nested": True, "content": None},
    ]


@pytest.fixture
def sample_parse_cache():
    """Cache de parsing con dimensionalidad inyectada."""
    return {
        "dependency_cycles": [["A", "B", "A"]],
        "dimensionality": {
            "material": 5,
            "labor": 4,
            "equipment": 4,
        },
    }


@pytest.fixture
def sample_config():
    """Configuración general para el procesador táctico."""
    return {
        "tolerance": 0.05,
        "currency": "USD",
        "rounding": 2,
    }


@pytest.fixture
def mock_dataframe():
    """DataFrame simulado con método to_dict."""
    df = MagicMock()
    df.to_dict.return_value = [
        {"col1": 1, "col2": 2},
        {"col1": 3, "col2": 4},
    ]
    return df


# =============================================================================
# 1. CONTRATOS Y ESTRUCTURAS DE DATOS
# =============================================================================
class TestVectorResultStatus:
    """Verifica la enumeración de estados."""

    def test_all_statuses_exist(self):
        assert VectorResultStatus.SUCCESS.value == "success"
        assert VectorResultStatus.PHYSICS_ERROR.value == "physics_error"
        assert VectorResultStatus.LOGIC_ERROR.value == "logic_error"
        assert VectorResultStatus.TOPOLOGY_ERROR.value == "topology_error"

    def test_status_count(self):
        assert len(VectorResultStatus) == 4

    def test_unique_values(self):
        values = [s.value for s in VectorResultStatus]
        assert len(values) == len(set(values))


class TestVectorMetrics:
    """Verifica la inmutabilidad y valores por defecto de VectorMetrics."""

    def test_default_values(self):
        m = VectorMetrics()
        assert m.processing_time_ms == 0.0
        assert m.memory_usage_mb == 0.0
        assert m.topological_coherence == 1.0
        assert m.algebraic_integrity == 1.0

    def test_custom_values(self):
        m = VectorMetrics(
            processing_time_ms=123.4,
            memory_usage_mb=56.7,
            topological_coherence=0.85,
            algebraic_integrity=0.92,
        )
        assert m.processing_time_ms == 123.4
        assert m.memory_usage_mb == 56.7
        assert m.topological_coherence == 0.85
        assert m.algebraic_integrity == 0.92

    def test_frozen_immutability(self):
        m = VectorMetrics(processing_time_ms=10.0)
        with pytest.raises(FrozenInstanceError):
            m.processing_time_ms = 20.0

    def test_frozen_no_new_attributes(self):
        m = VectorMetrics()
        with pytest.raises(FrozenInstanceError):
            m.new_field = "boom"

    def test_equality(self):
        m1 = VectorMetrics(processing_time_ms=10.0, memory_usage_mb=5.0)
        m2 = VectorMetrics(processing_time_ms=10.0, memory_usage_mb=5.0)
        assert m1 == m2

    def test_inequality(self):
        m1 = VectorMetrics(processing_time_ms=10.0)
        m2 = VectorMetrics(processing_time_ms=20.0)
        assert m1 != m2


# =============================================================================
# 2. HELPERS INTERNOS
# =============================================================================
class TestMeasureMemoryMb:
    """Pruebas para _measure_memory_mb con y sin psutil."""

    @patch("app.adapters.mic_vectors._HAS_PSUTIL", True)
    @patch("app.adapters.mic_vectors.psutil")
    def test_with_psutil(self, mock_psutil):
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 104_857_600  # 100 MB
        mock_psutil.Process.return_value = mock_process

        result = _measure_memory_mb()
        assert result == pytest.approx(100.0)

    @patch("app.adapters.mic_vectors._HAS_PSUTIL", False)
    def test_without_psutil(self):
        result = _measure_memory_mb()
        assert result == 0.0

    @patch("app.adapters.mic_vectors._HAS_PSUTIL", True)
    @patch("app.adapters.mic_vectors.psutil")
    def test_returns_float(self, mock_psutil):
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 52_428_800  # 50 MB
        mock_psutil.Process.return_value = mock_process

        result = _measure_memory_mb()
        assert isinstance(result, float)


class TestElapsedMs:
    """Pruebas para _elapsed_ms."""

    def test_positive_elapsed(self):
        start = time.time() - 0.1  # 100ms atrás
        result = _elapsed_ms(start)
        assert result >= 90.0  # al menos 90ms (margen)
        assert result < 500.0  # no más de 500ms

    def test_zero_elapsed(self):
        start = time.time()
        result = _elapsed_ms(start)
        assert result >= 0.0
        assert result < 50.0  # prácticamente instantáneo

    def test_returns_float(self):
        result = _elapsed_ms(time.time())
        assert isinstance(result, float)


class TestBuildResult:
    """Pruebas para _build_result — constructor canónico."""

    def test_success_result_has_required_keys(self):
        r = _build_result(
            success=True,
            stratum=Stratum.PHYSICS,
            status=VectorResultStatus.SUCCESS,
        )
        assert "success" in r
        assert "stratum" in r
        assert "status" in r
        assert "metrics" in r

    def test_success_result_no_error_key(self):
        r = _build_result(
            success=True,
            stratum=Stratum.PHYSICS,
            status=VectorResultStatus.SUCCESS,
        )
        assert "error" not in r

    def test_error_result_has_error_key(self):
        r = _build_result(
            success=False,
            stratum=Stratum.PHYSICS,
            status=VectorResultStatus.PHYSICS_ERROR,
            error="algo falló",
        )
        assert r["error"] == "algo falló"

    def test_payload_is_included(self):
        r = _build_result(
            success=True,
            stratum=Stratum.TACTICS,
            status=VectorResultStatus.SUCCESS,
            custom_data=[1, 2, 3],
            extra_info="hello",
        )
        assert r["custom_data"] == [1, 2, 3]
        assert r["extra_info"] == "hello"

    def test_default_metrics_when_none(self):
        r = _build_result(
            success=True,
            stratum=Stratum.PHYSICS,
            status=VectorResultStatus.SUCCESS,
        )
        expected_defaults = VectorMetrics().__dict__
        # Usamos asdict internamente, verificamos claves
        for key in expected_defaults:
            assert key in r["metrics"]

    def test_custom_metrics(self):
        m = VectorMetrics(processing_time_ms=42.0, memory_usage_mb=10.0)
        r = _build_result(
            success=True,
            stratum=Stratum.PHYSICS,
            status=VectorResultStatus.SUCCESS,
            metrics=m,
        )
        assert r["metrics"]["processing_time_ms"] == 42.0
        assert r["metrics"]["memory_usage_mb"] == 10.0

    def test_status_is_string_value(self):
        r = _build_result(
            success=True,
            stratum=Stratum.PHYSICS,
            status=VectorResultStatus.SUCCESS,
        )
        assert r["status"] == "success"


class TestBuildError:
    """Pruebas para _build_error — atajo de fallo."""

    def test_error_result_is_failure(self):
        r = _build_error(
            Stratum.PHYSICS,
            VectorResultStatus.PHYSICS_ERROR,
            "fallo catastrófico",
        )
        assert r["success"] is False

    def test_error_message_preserved(self):
        r = _build_error(
            Stratum.TACTICS,
            VectorResultStatus.LOGIC_ERROR,
            "registro vacío",
        )
        assert r["error"] == "registro vacío"

    def test_elapsed_time_in_metrics(self):
        r = _build_error(
            Stratum.PHYSICS,
            VectorResultStatus.TOPOLOGY_ERROR,
            "archivo no existe",
            elapsed_ms=123.45,
        )
        assert r["metrics"]["processing_time_ms"] == 123.45

    def test_default_elapsed_zero(self):
        r = _build_error(
            Stratum.PHYSICS,
            VectorResultStatus.PHYSICS_ERROR,
            "error",
        )
        assert r["metrics"]["processing_time_ms"] == 0.0


# =============================================================================
# 3. GUARDAS TOPOLÓGICAS
# =============================================================================
class TestValidateTopologicalPreconditions:
    """Pruebas para validate_topological_preconditions."""

    def test_valid_preconditions(self, tmp_file):
        config = {"key_a": 1, "key_b": 2}
        is_valid, err = validate_topological_preconditions(
            tmp_file, config, ["key_a", "key_b"]
        )
        assert is_valid is True
        assert err is None

    def test_file_not_exists(self, nonexistent_file):
        is_valid, err = validate_topological_preconditions(
            nonexistent_file, {}, []
        )
        assert is_valid is False
        assert "no existe" in err

    def test_missing_required_keys(self, tmp_file):
        config = {"key_a": 1}
        is_valid, err = validate_topological_preconditions(
            tmp_file, config, ["key_a", "key_b", "key_c"]
        )
        assert is_valid is False
        assert "key_b" in err
        assert "key_c" in err

    def test_non_numeric_dimension_constraints(self, tmp_file):
        config = {
            "dimension_constraints": {"x": "not_a_number", "y": 3}
        }
        is_valid, err = validate_topological_preconditions(
            tmp_file, config, []
        )
        assert is_valid is False
        assert "no numérica" in err

    def test_valid_dimension_constraints(self, tmp_file):
        config = {
            "dimension_constraints": {"x": 1, "y": 2.5, "z": 0}
        }
        is_valid, err = validate_topological_preconditions(
            tmp_file, config, []
        )
        assert is_valid is True
        assert err is None

    def test_no_dimension_constraints_is_ok(self, tmp_file):
        config = {"some_key": "value"}
        is_valid, err = validate_topological_preconditions(
            tmp_file, config, ["some_key"]
        )
        assert is_valid is True

    def test_empty_required_keys_always_passes_structure(self, tmp_file):
        is_valid, err = validate_topological_preconditions(
            tmp_file, {}, []
        )
        assert is_valid is True


class TestValidateHomologicalConstraints:
    """Pruebas para validate_homological_constraints."""

    def test_valid_constraints(self, valid_topological_constraints):
        assert validate_homological_constraints(
            valid_topological_constraints
        ) is True

    def test_missing_max_dimension(self):
        c = {"allow_holes": True, "connectivity": 0.9}
        assert validate_homological_constraints(c) is False

    def test_missing_allow_holes(self):
        c = {"max_dimension": 2, "connectivity": 0.9}
        assert validate_homological_constraints(c) is False

    def test_missing_connectivity(self):
        c = {"max_dimension": 2, "allow_holes": True}
        assert validate_homological_constraints(c) is False

    def test_negative_max_dimension(self):
        c = {"max_dimension": -1, "allow_holes": True, "connectivity": 0.9}
        assert validate_homological_constraints(c) is False

    def test_non_int_max_dimension(self):
        c = {"max_dimension": 2.5, "allow_holes": True, "connectivity": 0.9}
        assert validate_homological_constraints(c) is False

    def test_non_bool_allow_holes(self):
        c = {"max_dimension": 2, "allow_holes": "yes", "connectivity": 0.9}
        assert validate_homological_constraints(c) is False

    def test_non_numeric_connectivity(self):
        c = {"max_dimension": 2, "allow_holes": True, "connectivity": "high"}
        assert validate_homological_constraints(c) is False

    def test_zero_max_dimension_is_valid(self):
        c = {"max_dimension": 0, "allow_holes": False, "connectivity": 1.0}
        assert validate_homological_constraints(c) is True

    def test_empty_dict(self):
        assert validate_homological_constraints({}) is False


# =============================================================================
# 4. FUNCIONES DE TOPOLOGÍA ALGEBRAICA
# =============================================================================
class TestCalculateTopologicalCoherence:
    """
    Pruebas de la fórmula C = S·R / (1 + H).
    Verifica propiedades algebraicas: acotación, monotonicidad, absorción.
    """

    def test_empty_report_returns_zero(self):
        assert calculate_topological_coherence({}) == 0.0
        assert calculate_topological_coherence(None) == 0.0

    def test_perfect_coherence(self):
        report = {
            "stability_index": 1.0,
            "entropy": 0.0,
            "resonance_factor": 1.0,
        }
        result = calculate_topological_coherence(report)
        assert result == pytest.approx(1.0)

    def test_zero_stability_absorbs(self):
        report = {
            "stability_index": 0.0,
            "entropy": 0.0,
            "resonance_factor": 1.0,
        }
        assert calculate_topological_coherence(report) == 0.0

    def test_zero_resonance_absorbs(self):
        report = {
            "stability_index": 1.0,
            "entropy": 0.0,
            "resonance_factor": 0.0,
        }
        assert calculate_topological_coherence(report) == 0.0

    def test_bounded_zero_one(self):
        """C ∈ [0, 1] para cualquier entrada razonable."""
        cases = [
            {"stability_index": 0.5, "entropy": 0.3, "resonance_factor": 0.7},
            {"stability_index": 1.0, "entropy": 10.0, "resonance_factor": 1.0},
            {"stability_index": 0.0, "entropy": 0.0, "resonance_factor": 0.0},
            {"stability_index": 1.0, "entropy": 0.0, "resonance_factor": 1.0},
        ]
        for report in cases:
            c = calculate_topological_coherence(report)
            assert 0.0 <= c <= 1.0, f"Fuera de [0,1]: {c} para {report}"

    def test_monotone_increasing_stability(self):
        """∂C/∂S > 0: más estabilidad → más coherencia."""
        base = {"entropy": 0.5, "resonance_factor": 0.8}
        c_low = calculate_topological_coherence({**base, "stability_index": 0.3})
        c_high = calculate_topological_coherence({**base, "stability_index": 0.9})
        assert c_high > c_low

    def test_monotone_increasing_resonance(self):
        """∂C/∂R > 0: más resonancia → más coherencia."""
        base = {"stability_index": 0.7, "entropy": 0.5}
        c_low = calculate_topological_coherence({**base, "resonance_factor": 0.2})
        c_high = calculate_topological_coherence({**base, "resonance_factor": 0.9})
        assert c_high > c_low

    def test_monotone_decreasing_entropy(self):
        """∂C/∂H < 0: más entropía → menos coherencia."""
        base = {"stability_index": 0.8, "resonance_factor": 0.9}
        c_low_entropy = calculate_topological_coherence({**base, "entropy": 0.1})
        c_high_entropy = calculate_topological_coherence({**base, "entropy": 5.0})
        assert c_low_entropy > c_high_entropy

    def test_specific_value(self):
        """Valor concreto: S=0.8, R=0.6, H=0.2 → C = 0.48/1.2 = 0.4."""
        report = {
            "stability_index": 0.8,
            "entropy": 0.2,
            "resonance_factor": 0.6,
        }
        expected = 0.8 * 0.6 / (1.0 + 0.2)  # 0.4
        assert calculate_topological_coherence(report) == pytest.approx(expected)

    def test_missing_keys_use_defaults(self):
        """Claves ausentes → defaults (S=0, H=0, R=0) → C=0."""
        result = calculate_topological_coherence({"unrelated": 42})
        assert result == 0.0

    def test_negative_values_clamped(self):
        """Valores negativos se fijan a 0 por los max()."""
        report = {
            "stability_index": -0.5,
            "entropy": -1.0,
            "resonance_factor": -0.3,
        }
        result = calculate_topological_coherence(report)
        assert result == 0.0  # S=0 absorbe


class TestCalculateBettiNumbers:
    """Pruebas de números de Betti y característica de Euler."""

    def test_empty_records(self):
        result = calculate_betti_numbers([], {})
        assert result == {"beta_0": 0, "beta_1": 0, "beta_2": 0, "euler": 0}

    def test_single_type_no_cycles(self):
        records = [
            {"record_type": "material", "value": 1},
            {"record_type": "material", "value": 2},
        ]
        result = calculate_betti_numbers(records, {})
        assert result["beta_0"] == 1  # un solo tipo
        assert result["beta_1"] == 0
        assert result["beta_2"] == 0
        assert result["euler"] == 1  # χ = 1 - 0 + 0

    def test_multiple_types(self, sample_raw_records):
        result = calculate_betti_numbers(sample_raw_records, {})
        # material, labor, equipment = 3 tipos
        assert result["beta_0"] == 3

    def test_cycles_from_cache(self, sample_raw_records, sample_parse_cache):
        result = calculate_betti_numbers(sample_raw_records, sample_parse_cache)
        assert result["beta_1"] == 1  # un ciclo en cache

    def test_cavities_nested_empty(self, sample_raw_records):
        # El último registro tiene nested=True, content=None
        result = calculate_betti_numbers(sample_raw_records, {})
        assert result["beta_2"] == 1

    def test_euler_characteristic(self, sample_raw_records, sample_parse_cache):
        result = calculate_betti_numbers(sample_raw_records, sample_parse_cache)
        expected_euler = result["beta_0"] - result["beta_1"] + result["beta_2"]
        assert result["euler"] == expected_euler

    def test_no_dependency_cycles_key(self):
        records = [{"record_type": "A"}]
        result = calculate_betti_numbers(records, {"other": "data"})
        assert result["beta_1"] == 0

    def test_non_list_cycles_treated_as_zero(self):
        records = [{"record_type": "A"}]
        cache = {"dependency_cycles": "not_a_list"}
        result = calculate_betti_numbers(records, cache)
        assert result["beta_1"] == 0

    def test_unknown_record_type_default(self):
        records = [{"value": 1}, {"value": 2}]  # sin record_type
        result = calculate_betti_numbers(records, {})
        assert result["beta_0"] == 1  # ambos son "unknown"

    def test_all_nested_with_content(self):
        records = [
            {"record_type": "A", "nested": True, "content": "data"},
            {"record_type": "A", "nested": True, "content": "more"},
        ]
        result = calculate_betti_numbers(records, {})
        assert result["beta_2"] == 0  # tienen contenido


class TestCalculateAlgebraicIntegrity:
    """Pruebas de I = 1/(1 + Σβ_i, i>0)."""

    def test_empty_dict(self):
        assert calculate_algebraic_integrity({}) == 1.0

    def test_no_higher_betti(self):
        betti = {"beta_0": 5, "euler": 5}
        assert calculate_algebraic_integrity(betti) == 1.0

    def test_with_cycles(self):
        betti = {"beta_0": 3, "beta_1": 2, "beta_2": 0, "euler": 1}
        # higher = beta_1 + beta_2 = 2
        assert calculate_algebraic_integrity(betti) == pytest.approx(1.0 / 3.0)

    def test_with_cycles_and_cavities(self):
        betti = {"beta_0": 3, "beta_1": 2, "beta_2": 3, "euler": 4}
        # higher = 2 + 3 = 5
        assert calculate_algebraic_integrity(betti) == pytest.approx(1.0 / 6.0)

    def test_only_beta_0(self):
        betti = {"beta_0": 10}
        assert calculate_algebraic_integrity(betti) == 1.0

    def test_integrity_decreases_with_defects(self):
        """Más defectos topológicos → menor integridad."""
        i_clean = calculate_algebraic_integrity(
            {"beta_0": 3, "beta_1": 0, "beta_2": 0, "euler": 3}
        )
        i_dirty = calculate_algebraic_integrity(
            {"beta_0": 3, "beta_1": 5, "beta_2": 3, "euler": 1}
        )
        assert i_clean > i_dirty

    def test_integrity_bounded_zero_one(self):
        """I ∈ (0, 1] para cualquier entrada."""
        cases = [
            {},
            {"beta_0": 1},
            {"beta_0": 1, "beta_1": 100, "beta_2": 200, "euler": 101},
        ]
        for betti in cases:
            i = calculate_algebraic_integrity(betti)
            assert 0.0 < i <= 1.0


class TestCalculateDimensionality:
    """Pruebas de dim(V_t) = |⋃ keys(r) : type(r)=t|."""

    def test_empty_records(self):
        assert calculate_dimensionality([]) == {}

    def test_single_record(self):
        records = [{"record_type": "A", "x": 1, "y": 2}]
        result = calculate_dimensionality(records)
        assert result == {"A": 3}  # record_type, x, y

    def test_union_of_keys(self):
        """Dos registros del mismo tipo con claves distintas → unión."""
        records = [
            {"record_type": "A", "x": 1},
            {"record_type": "A", "y": 2},
        ]
        result = calculate_dimensionality(records)
        # Unión: {record_type, x, y} = 3
        assert result["A"] == 3

    def test_same_keys_no_inflation(self):
        """Registros con mismas claves no inflan la dimensión."""
        records = [
            {"record_type": "A", "x": 1, "y": 2},
            {"record_type": "A", "x": 3, "y": 4},
            {"record_type": "A", "x": 5, "y": 6},
        ]
        result = calculate_dimensionality(records)
        assert result["A"] == 3  # {record_type, x, y}

    def test_multiple_types(self, sample_raw_records):
        result = calculate_dimensionality(sample_raw_records)
        assert "material" in result
        assert "labor" in result
        assert "equipment" in result

    def test_default_type_when_missing(self):
        records = [{"x": 1}, {"y": 2}]
        result = calculate_dimensionality(records)
        assert "default" in result
        assert result["default"] == 3  # {x, y, record_type no está} → {x, y}=2... wait

    def test_default_type_union(self):
        """Sin record_type explícito, se agrupan como 'default'."""
        records = [{"a": 1, "b": 2}, {"b": 3, "c": 4}]
        result = calculate_dimensionality(records)
        # Unión: {a, b, c} = 3
        assert result["default"] == 3


class TestValidateDimensionalIsomorphism:
    """Pruebas del isomorfismo dimensional con tolerancia ε."""

    def test_both_empty_vacuously_true(self):
        assert validate_dimensional_isomorphism({}, {}) is True

    def test_expected_empty_vacuously_true(self):
        assert validate_dimensional_isomorphism({}, {"A": 5}) is True

    def test_actual_empty_vacuously_true(self):
        assert validate_dimensional_isomorphism({"A": 5}, {}) is True

    def test_identical_dimensions(self):
        d = {"A": 10, "B": 20}
        assert validate_dimensional_isomorphism(d, d) is True

    def test_within_tolerance(self):
        expected = {"A": 100, "B": 200}
        actual = {"A": 105, "B": 210}  # 5% y 5%
        assert validate_dimensional_isomorphism(expected, actual) is True

    def test_beyond_tolerance(self):
        expected = {"A": 100}
        actual = {"A": 115}  # 15% > 10%
        assert validate_dimensional_isomorphism(expected, actual) is False

    def test_different_keys(self):
        expected = {"A": 10}
        actual = {"B": 10}
        assert validate_dimensional_isomorphism(expected, actual) is False

    def test_custom_tolerance(self):
        expected = {"A": 100}
        actual = {"A": 120}

        # Con tolerancia 10% → fallo (20%)
        assert validate_dimensional_isomorphism(
            expected, actual, tolerance=0.10
        ) is False

        # Con tolerancia 25% → éxito (20% < 25%)
        assert validate_dimensional_isomorphism(
            expected, actual, tolerance=0.25
        ) is True

    def test_zero_expected_avoids_division_by_zero(self):
        """max(0, 1) = 1 evita ZeroDivisionError."""
        expected = {"A": 0}
        actual = {"A": 0}
        assert validate_dimensional_isomorphism(expected, actual) is True


# =============================================================================
# 5. VECTOR FÍSICO 1: ESTABILIZACIÓN DE FLUJO
# =============================================================================
class TestVectorStabilizeFlux:
    """Pruebas del vector de estabilización de flujo."""

    def test_file_not_exists(self, nonexistent_file, valid_physics_config):
        result = vector_stabilize_flux(nonexistent_file, valid_physics_config)
        assert result["success"] is False
        assert result["status"] == "topology_error"
        assert "no existe" in result["error"]

    def test_missing_config_keys(self, tmp_file):
        config = {"system_capacitance": 1.0}  # faltan 2 claves
        result = vector_stabilize_flux(tmp_file, config)
        assert result["success"] is False
        assert result["status"] == "topology_error"

    @patch("app.adapters.mic_vectors.DataFluxCondenser")
    @patch("app.adapters.mic_vectors.CondenserConfig")
    def test_successful_stabilization(
        self, MockConfig, MockCondenser, tmp_file, valid_physics_config, mock_dataframe
    ):
        mock_instance = MagicMock()
        mock_instance.stabilize.return_value = mock_dataframe
        mock_instance.get_physics_report.return_value = {
            "stability_index": 0.9,
            "entropy": 0.1,
            "resonance_factor": 0.8,
        }
        MockCondenser.return_value = mock_instance

        result = vector_stabilize_flux(tmp_file, valid_physics_config)

        assert result["success"] is True
        assert result["status"] == "success"
        assert result["stratum"] == Stratum.PHYSICS
        assert "data" in result
        assert "physics_metrics" in result
        assert "metrics" in result

    @patch("app.adapters.mic_vectors.DataFluxCondenser")
    @patch("app.adapters.mic_vectors.CondenserConfig")
    def test_condenser_exception(
        self, MockConfig, MockCondenser, tmp_file, valid_physics_config
    ):
        MockCondenser.return_value.stabilize.side_effect = RuntimeError(
            "Resonancia crítica"
        )

        result = vector_stabilize_flux(tmp_file, valid_physics_config)
        assert result["success"] is False
        assert result["status"] == "physics_error"
        assert "Resonancia crítica" in result["error"]

    @patch("app.adapters.mic_vectors.DataFluxCondenser")
    @patch("app.adapters.mic_vectors.CondenserConfig")
    def test_metrics_include_processing_time(
        self, MockConfig, MockCondenser, tmp_file, valid_physics_config, mock_dataframe
    ):
        mock_instance = MagicMock()
        mock_instance.stabilize.return_value = mock_dataframe
        mock_instance.get_physics_report.return_value = {}
        MockCondenser.return_value = mock_instance

        result = vector_stabilize_flux(tmp_file, valid_physics_config)
        assert result["metrics"]["processing_time_ms"] >= 0

    @patch("app.adapters.mic_vectors.DataFluxCondenser")
    @patch("app.adapters.mic_vectors.CondenserConfig")
    def test_coherence_in_metrics(
        self, MockConfig, MockCondenser, tmp_file, valid_physics_config, mock_dataframe
    ):
        mock_instance = MagicMock()
        mock_instance.stabilize.return_value = mock_dataframe
        mock_instance.get_physics_report.return_value = {
            "stability_index": 0.9,
            "entropy": 0.1,
            "resonance_factor": 0.8,
        }
        MockCondenser.return_value = mock_instance

        result = vector_stabilize_flux(tmp_file, valid_physics_config)
        coherence = result["metrics"]["topological_coherence"]
        expected = 0.9 * 0.8 / 1.1
        assert coherence == pytest.approx(expected, rel=1e-6)


# =============================================================================
# 6. VECTOR FÍSICO 2: PARSING TOPOLÓGICO
# =============================================================================
class TestVectorParseRawStructure:
    """Pruebas del vector de parsing topológico."""

    def test_file_not_exists(self, nonexistent_file, valid_profile):
        result = vector_parse_raw_structure(nonexistent_file, valid_profile)
        assert result["success"] is False
        assert result["status"] == "topology_error"

    def test_invalid_topological_constraints(self, tmp_file, valid_profile):
        bad_constraints = {"max_dimension": -1, "allow_holes": "yes"}
        result = vector_parse_raw_structure(
            tmp_file, valid_profile, bad_constraints
        )
        assert result["success"] is False
        assert result["status"] == "topology_error"
        assert "homológicas" in result["error"]

    @patch("app.adapters.mic_vectors.ReportParserCrudo")
    def test_successful_parsing(
        self, MockParser, tmp_file, valid_profile, sample_raw_records
    ):
        mock_instance = MagicMock()
        mock_instance.parse_to_raw.return_value = sample_raw_records
        mock_instance.get_parse_cache.return_value = {"dependency_cycles": []}
        mock_instance.validation_stats = MagicMock(
            __dict__={"valid": 5, "invalid": 0}
        )
        MockParser.return_value = mock_instance

        result = vector_parse_raw_structure(tmp_file, valid_profile)

        assert result["success"] is True
        assert result["status"] == "success"
        assert "raw_records" in result
        assert "parse_cache" in result
        assert "homological_invariants" in result
        assert "validation_stats" in result

    @patch("app.adapters.mic_vectors.ReportParserCrudo")
    def test_cache_receives_dimensionality(
        self, MockParser, tmp_file, valid_profile, sample_raw_records
    ):
        """Verifica que dimensionality se inyecta en el cache."""
        mock_instance = MagicMock()
        mock_instance.parse_to_raw.return_value = sample_raw_records
        mock_instance.get_parse_cache.return_value = {}
        mock_instance.validation_stats = MagicMock(__dict__={})
        MockParser.return_value = mock_instance

        result = vector_parse_raw_structure(tmp_file, valid_profile)
        cache = result["parse_cache"]
        assert "dimensionality" in cache
        assert isinstance(cache["dimensionality"], dict)

    @patch("app.adapters.mic_vectors.ReportParserCrudo")
    def test_parser_exception(self, MockParser, tmp_file, valid_profile):
        MockParser.return_value.parse_to_raw.side_effect = ValueError(
            "Formato corrupto"
        )

        result = vector_parse_raw_structure(tmp_file, valid_profile)
        assert result["success"] is False
        assert result["status"] == "physics_error"
        assert "Formato corrupto" in result["error"]

    @patch("app.adapters.mic_vectors.ReportParserCrudo")
    def test_homological_invariants_structure(
        self, MockParser, tmp_file, valid_profile
    ):
        mock_instance = MagicMock()
        mock_instance.parse_to_raw.return_value = [
            {"record_type": "A", "v": 1},
            {"record_type": "B", "v": 2},
        ]
        mock_instance.get_parse_cache.return_value = {
            "dependency_cycles": [["x", "y", "x"]]
        }
        mock_instance.validation_stats = MagicMock(__dict__={})
        MockParser.return_value = mock_instance

        result = vector_parse_raw_structure(tmp_file, valid_profile)
        inv = result["homological_invariants"]
        assert "beta_0" in inv
        assert "beta_1" in inv
        assert "beta_2" in inv
        assert "euler" in inv
        assert inv["beta_0"] == 2  # tipos A, B
        assert inv["beta_1"] == 1  # un ciclo

    @patch("app.adapters.mic_vectors.ReportParserCrudo")
    def test_valid_topological_constraints_passed(
        self, MockParser, tmp_file, valid_profile, valid_topological_constraints
    ):
        mock_instance = MagicMock()
        mock_instance.parse_to_raw.return_value = [{"record_type": "A"}]
        mock_instance.get_parse_cache.return_value = {}
        mock_instance.validation_stats = MagicMock(__dict__={})
        MockParser.return_value = mock_instance

        result = vector_parse_raw_structure(
            tmp_file, valid_profile, valid_topological_constraints
        )
        assert result["success"] is True

    @patch("app.adapters.mic_vectors.ReportParserCrudo")
    def test_no_validation_stats_attribute(
        self, MockParser, tmp_file, valid_profile
    ):
        """Parser sin validation_stats no debe fallar."""
        mock_instance = MagicMock(spec=[])
        mock_instance.parse_to_raw = MagicMock(
            return_value=[{"record_type": "A"}]
        )
        mock_instance.get_parse_cache = MagicMock(return_value={})
        # No tiene validation_stats
        MockParser.return_value = mock_instance

        result = vector_parse_raw_structure(tmp_file, valid_profile)
        assert result["success"] is True
        assert result["validation_stats"] == {}


# =============================================================================
# 7. VECTOR TÁCTICO: ESTRUCTURACIÓN LÓGICA
# =============================================================================
class TestVectorStructureLogic:
    """Pruebas del vector táctico."""

    def test_empty_records_returns_error(self, sample_parse_cache, sample_config):
        result = vector_structure_logic([], sample_parse_cache, sample_config)
        assert result["success"] is False
        assert result["status"] == "logic_error"
        assert "vacío" in result["error"]

    @patch("app.adapters.mic_vectors.APUProcessor")
    def test_successful_processing(
        self,
        MockProcessor,
        sample_raw_records,
        sample_parse_cache,
        sample_config,
        mock_dataframe,
    ):
        mock_instance = MagicMock()
        mock_instance.process_all.return_value = mock_dataframe
        mock_instance.get_validation_kernel.return_value = {"errors": []}
        mock_instance.get_topological_coherence.return_value = 0.95
        mock_instance.get_algebraic_integrity.return_value = 0.88
        mock_instance.get_quality_report.return_value = {"score": 92}
        MockProcessor.return_value = mock_instance

        result = vector_structure_logic(
            sample_raw_records, sample_parse_cache, sample_config
        )

        assert result["success"] is True
        assert result["status"] == "success"
        assert result["stratum"] == Stratum.TACTICS
        assert "processed_data" in result
        assert "quality_report" in result
        assert "algebraic_kernel" in result

    @patch("app.adapters.mic_vectors.APUProcessor")
    def test_processor_exception(
        self,
        MockProcessor,
        sample_raw_records,
        sample_parse_cache,
        sample_config,
    ):
        MockProcessor.return_value.process_all.side_effect = RuntimeError(
            "Overflow en APU"
        )

        result = vector_structure_logic(
            sample_raw_records, sample_parse_cache, sample_config
        )
        assert result["success"] is False
        assert result["status"] == "logic_error"
        assert "Overflow en APU" in result["error"]

    @patch("app.adapters.mic_vectors.APUProcessor")
    def test_metrics_populated(
        self,
        MockProcessor,
        sample_raw_records,
        sample_parse_cache,
        sample_config,
        mock_dataframe,
    ):
        mock_instance = MagicMock()
        mock_instance.process_all.return_value = mock_dataframe
        mock_instance.get_validation_kernel.return_value = {}
        mock_instance.get_topological_coherence.return_value = 0.85
        mock_instance.get_algebraic_integrity.return_value = 0.90
        mock_instance.get_quality_report.return_value = {}
        MockProcessor.return_value = mock_instance

        result = vector_structure_logic(
            sample_raw_records, sample_parse_cache, sample_config
        )
        m = result["metrics"]
        assert m["processing_time_ms"] >= 0
        assert m["topological_coherence"] == 0.85
        assert m["algebraic_integrity"] == 0.90

    @patch("app.adapters.mic_vectors.APUProcessor")
    def test_missing_optional_methods_fallback(
        self,
        MockProcessor,
        sample_raw_records,
        sample_parse_cache,
        sample_config,
        mock_dataframe,
    ):
        """Processor sin métodos opcionales → fallback a defaults."""
        mock_instance = MagicMock(spec=[])
        mock_instance.process_all = MagicMock(return_value=mock_dataframe)
        mock_instance.raw_records = None
        # No tiene get_validation_kernel, get_topological_coherence, etc.
        MockProcessor.return_value = mock_instance

        result = vector_structure_logic(
            sample_raw_records, sample_parse_cache, sample_config
        )
        assert result["success"] is True
        assert result["algebraic_kernel"] == {}
        assert result["quality_report"] == {}
        assert result["metrics"]["topological_coherence"] == 1.0
        assert result["metrics"]["algebraic_integrity"] == 1.0

    @patch("app.adapters.mic_vectors.APUProcessor")
    def test_dimensional_isomorphism_warning(
        self,
        MockProcessor,
        sample_raw_records,
        sample_config,
        mock_dataframe,
        caplog,
    ):
        """Isomorfismo roto produce warning, no fallo."""
        incompatible_cache = {
            "dimensionality": {"material": 100, "labor": 100, "equipment": 100}
        }
        mock_instance = MagicMock()
        mock_instance.process_all.return_value = mock_dataframe
        mock_instance.get_validation_kernel.return_value = {}
        mock_instance.get_topological_coherence.return_value = 1.0
        mock_instance.get_algebraic_integrity.return_value = 1.0
        mock_instance.get_quality_report.return_value = {}
        MockProcessor.return_value = mock_instance

        with caplog.at_level("WARNING"):
            result = vector_structure_logic(
                sample_raw_records, incompatible_cache, sample_config
            )

        assert result["success"] is True
        assert any("Isomorfismo dimensional roto" in r.message for r in caplog.records)

    @patch("app.adapters.mic_vectors.APUProcessor")
    def test_algebraic_structure_passed_to_processor(
        self,
        MockProcessor,
        sample_raw_records,
        sample_parse_cache,
        sample_config,
        mock_dataframe,
    ):
        mock_instance = MagicMock()
        mock_instance.process_all.return_value = mock_dataframe
        mock_instance.get_validation_kernel.return_value = {}
        mock_instance.get_topological_coherence.return_value = 1.0
        mock_instance.get_algebraic_integrity.return_value = 1.0
        mock_instance.get_quality_report.return_value = {}
        MockProcessor.return_value = mock_instance

        vector_structure_logic(
            sample_raw_records,
            sample_parse_cache,
            sample_config,
            algebraic_structure="ring",
        )

        MockProcessor.assert_called_once_with(
            config=sample_config,
            parse_cache=sample_parse_cache,
            algebraic_structure="ring",
        )


# =============================================================================
# 8. VECTOR FACTORY
# =============================================================================
class TestVectorFactory:
    """Pruebas de la fábrica de vectores."""

    def setup_method(self):
        """Reset del registry antes de cada test."""
        VectorFactory.reset_registry()

    def test_create_known_physics_vector(self):
        wrapper = VectorFactory.create_physics_vector("stabilize")
        assert callable(wrapper)
        assert wrapper.__name__ == "physics_stabilize"

    def test_create_parse_vector(self):
        wrapper = VectorFactory.create_physics_vector("parse")
        assert callable(wrapper)
        assert wrapper.__name__ == "physics_parse"

    def test_create_unknown_vector_raises(self):
        with pytest.raises(ValueError, match="no registrado"):
            VectorFactory.create_physics_vector("warp_drive")

    def test_error_message_lists_available(self):
        with pytest.raises(ValueError, match="parse"):
            VectorFactory.create_physics_vector("nonexistent")

    def test_create_tactics_vector(self):
        wrapper = VectorFactory.create_tactics_vector()
        assert callable(wrapper)
        assert wrapper.__name__ == "tactics_structure"

    def test_register_custom_vector(self):
        custom_fn = MagicMock(return_value={"success": True})
        VectorFactory.register_physics_vector("custom", custom_fn)

        wrapper = VectorFactory.create_physics_vector("custom")
        result = wrapper("file.csv")
        assert result["success"] is True

    def test_reset_removes_custom(self):
        custom_fn = MagicMock()
        VectorFactory.register_physics_vector("custom", custom_fn)

        VectorFactory.reset_registry()

        with pytest.raises(ValueError):
            VectorFactory.create_physics_vector("custom")

    def test_defaults_merged_into_kwargs(self):
        spy = MagicMock(return_value={"success": True})
        VectorFactory.register_physics_vector("spy", spy)

        wrapper = VectorFactory.create_physics_vector(
            "spy", default_param="value_a"
        )
        wrapper("file.csv", extra="value_b")

        spy.assert_called_once_with(
            "file.csv", default_param="value_a", extra="value_b"
        )

    def test_kwargs_override_defaults(self):
        spy = MagicMock(return_value={"success": True})
        VectorFactory.register_physics_vector("spy", spy)

        wrapper = VectorFactory.create_physics_vector(
            "spy", param="default"
        )
        wrapper("file.csv", param="override")

        spy.assert_called_once_with("file.csv", param="override")

    def test_tactics_defaults_merged(self):
        with patch(
            "app.adapters.mic_vectors.vector_structure_logic"
        ) as mock_vsl:
            mock_vsl.return_value = {"success": True}

            wrapper = VectorFactory.create_tactics_vector(
                algebraic_structure="ring"
            )
            wrapper([], {}, {})

            mock_vsl.assert_called_once_with(
                [], {}, {}, algebraic_structure="ring"
            )

    def test_wrapper_preserves_docstring(self):
        wrapper = VectorFactory.create_physics_vector("stabilize")
        assert wrapper.__doc__ is not None
        assert "PHYSICS" in wrapper.__doc__

    def test_tactics_wrapper_preserves_docstring(self):
        wrapper = VectorFactory.create_tactics_vector()
        assert wrapper.__doc__ is not None
        assert "TACTICS" in wrapper.__doc__


# =============================================================================
# 9. COMPOSICIÓN DE VECTORES
# =============================================================================
class TestComposeVectors:
    """Pruebas del pipeline de composición φ_tact ∘ φ_phys."""

    def test_physics_failure_short_circuits(self):
        physics_fn = MagicMock(
            return_value={
                "success": False,
                "error": "archivo corrupto",
                "status": "physics_error",
                "stratum": Stratum.PHYSICS,
                "metrics": {},
            }
        )
        tactics_fn = MagicMock()

        result = compose_vectors(
            physics_fn, tactics_fn, ("file.csv",), {}
        )

        assert result["success"] is False
        tactics_fn.assert_not_called()

    def test_undefined_composition_no_raw_records(self):
        """Vector físico que no produce raw_records → composición indefinida."""
        physics_fn = MagicMock(
            return_value={
                "success": True,
                "data": [{"col": 1}],  # tiene data, no raw_records
                "status": "success",
                "stratum": Stratum.PHYSICS,
                "metrics": {},
            }
        )
        tactics_fn = MagicMock()

        result = compose_vectors(
            physics_fn, tactics_fn, ("file.csv",), {}
        )

        assert result["success"] is False
        assert "Composición indefinida" in result["error"]
        tactics_fn.assert_not_called()

    def test_successful_composition(self):
        physics_fn = MagicMock(
            return_value={
                "success": True,
                "raw_records": [{"record_type": "A"}],
                "parse_cache": {"key": "val"},
                "metrics": {
                    "processing_time_ms": 50.0,
                    "topological_coherence": 0.9,
                },
            }
        )
        tactics_fn = MagicMock(
            return_value={
                "success": True,
                "processed_data": [{"out": 1}],
                "metrics": {
                    "processing_time_ms": 30.0,
                    "topological_coherence": 0.85,
                },
            }
        )

        config = {"param": "value"}
        result = compose_vectors(
            physics_fn, tactics_fn, ("file.csv", {}), config
        )

        assert result["success"] is True
        assert "combined_metrics" in result

        # Verifica métricas combinadas
        cm = result["combined_metrics"]
        assert cm["total_time_ms"] == pytest.approx(80.0)
        assert cm["pipeline_coherence"] == pytest.approx(0.85)  # min(0.9, 0.85)

    def test_tactics_receives_raw_records_and_cache(self):
        raw = [{"record_type": "X", "val": 42}]
        cache = {"dim": 3}

        physics_fn = MagicMock(
            return_value={
                "success": True,
                "raw_records": raw,
                "parse_cache": cache,
                "metrics": {},
            }
        )
        tactics_fn = MagicMock(
            return_value={"success": True, "metrics": {}}
        )

        config = {"mode": "fast"}
        compose_vectors(physics_fn, tactics_fn, ("file.csv",), config)

        tactics_fn.assert_called_once_with(raw, cache, config)

    def test_pipeline_coherence_is_minimum(self):
        """pipeline_coherence = min(phys, tact) — cuello de botella."""
        physics_fn = MagicMock(
            return_value={
                "success": True,
                "raw_records": [{}],
                "parse_cache": {},
                "metrics": {"topological_coherence": 0.5, "processing_time_ms": 10},
            }
        )
        tactics_fn = MagicMock(
            return_value={
                "success": True,
                "metrics": {"topological_coherence": 0.99, "processing_time_ms": 5},
            }
        )

        result = compose_vectors(physics_fn, tactics_fn, ("f",), {})
        assert result["combined_metrics"]["pipeline_coherence"] == pytest.approx(0.5)

    def test_tactics_failure_propagated(self):
        physics_fn = MagicMock(
            return_value={
                "success": True,
                "raw_records": [{}],
                "parse_cache": {},
                "metrics": {},
            }
        )
        tactics_fn = MagicMock(
            return_value={
                "success": False,
                "error": "lógica rota",
                "status": "logic_error",
                "metrics": {},
            }
        )

        result = compose_vectors(physics_fn, tactics_fn, ("f",), {})
        assert result["success"] is False
        assert "combined_metrics" in result  # aún se calculan métricas

    def test_empty_parse_cache_default(self):
        """Si physics no produce parse_cache, se usa {} vacío."""
        physics_fn = MagicMock(
            return_value={
                "success": True,
                "raw_records": [{"r": 1}],
                # sin parse_cache
                "metrics": {},
            }
        )
        tactics_fn = MagicMock(
            return_value={"success": True, "metrics": {}}
        )

        compose_vectors(physics_fn, tactics_fn, ("f",), {})

        # El segundo arg de tactics_fn debe ser {}
        call_args = tactics_fn.call_args
        assert call_args[0][1] == {}


# =============================================================================
# 10. PROPIEDADES ALGEBRAICAS TRANSVERSALES
# =============================================================================
class TestAlgebraicProperties:
    """
    Pruebas de propiedades matemáticas que deben sostenerse
    independientemente de los datos de entrada.
    """

    @pytest.mark.parametrize(
        "stability,entropy,resonance",
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 1.0),
            (0.5, 0.5, 0.5),
            (1.0, 100.0, 1.0),
            (0.3, 0.0, 0.7),
            (0.99, 0.01, 0.99),
        ],
    )
    def test_coherence_in_unit_interval(self, stability, entropy, resonance):
        """∀ S,H,R: C ∈ [0, 1]."""
        report = {
            "stability_index": stability,
            "entropy": entropy,
            "resonance_factor": resonance,
        }
        c = calculate_topological_coherence(report)
        assert 0.0 <= c <= 1.0

    @pytest.mark.parametrize(
        "betti",
        [
            {"beta_0": 1, "beta_1": 0, "beta_2": 0, "euler": 1},
            {"beta_0": 5, "beta_1": 3, "beta_2": 1, "euler": 3},
            {"beta_0": 10, "beta_1": 10, "beta_2": 10, "euler": 10},
            {},
        ],
    )
    def test_integrity_in_unit_interval(self, betti):
        """∀ β: I ∈ (0, 1]."""
        i = calculate_algebraic_integrity(betti)
        assert 0.0 < i <= 1.0

    def test_euler_characteristic_formula(self):
        """χ = β₀ − β₁ + β₂ (relación de Euler-Poincaré)."""
        records = [
            {"record_type": "A", "nested": True, "content": None},
            {"record_type": "B"},
            {"record_type": "A"},
        ]
        cache = {"dependency_cycles": [["x", "y"], ["a", "b"]]}

        betti = calculate_betti_numbers(records, cache)
        assert betti["euler"] == betti["beta_0"] - betti["beta_1"] + betti["beta_2"]

    def test_dimensionality_idempotent(self, sample_raw_records):
        """dim(records) aplicado dos veces produce el mismo resultado."""
        d1 = calculate_dimensionality(sample_raw_records)
        d2 = calculate_dimensionality(sample_raw_records)
        assert d1 == d2

    def test_isomorphism_reflexive(self):
        """∀ D: iso(D, D) = True (reflexividad)."""
        d = {"A": 10, "B": 20, "C": 30}
        assert validate_dimensional_isomorphism(d, d) is True

    def test_isomorphism_symmetric(self):
        """iso(D1, D2) ⟺ iso(D2, D1) (simetría) con mismas dimensiones."""
        d1 = {"A": 100, "B": 200}
        d2 = {"A": 100, "B": 200}
        assert validate_dimensional_isomorphism(d1, d2) == validate_dimensional_isomorphism(d2, d1)

    @pytest.mark.parametrize(
        "records,expected_beta0",
        [
            ([], 0),
            ([{"record_type": "A"}], 1),
            ([{"record_type": "A"}, {"record_type": "B"}], 2),
            (
                [
                    {"record_type": "A"},
                    {"record_type": "B"},
                    {"record_type": "C"},
                ],
                3,
            ),
        ],
    )
    def test_beta0_equals_unique_types(self, records, expected_beta0):
        """β₀ = |{tipos únicos}| siempre."""
        betti = calculate_betti_numbers(records, {})
        assert betti["beta_0"] == expected_beta0

    def test_build_result_always_has_canonical_keys(self):
        """Todo resultado canónico tiene las 4 claves obligatorias."""
        canonical_keys = {"success", "stratum", "status", "metrics"}

        for status in VectorResultStatus:
            r = _build_result(
                success=status == VectorResultStatus.SUCCESS,
                stratum=Stratum.PHYSICS,
                status=status,
            )
            assert canonical_keys.issubset(r.keys())

    def test_coherence_identity_element(self):
        """C(S=1, H=0, R=1) = 1 — elemento identidad de coherencia."""
        report = {
            "stability_index": 1.0,
            "entropy": 0.0,
            "resonance_factor": 1.0,
        }
        assert calculate_topological_coherence(report) == 1.0

    def test_integrity_identity_element(self):
        """I(β_higher=0) = 1 — integridad perfecta sin defectos."""
        betti = {"beta_0": 42, "euler": 42}
        assert calculate_algebraic_integrity(betti) == 1.0

    def test_composition_preserves_physics_context_on_undefined(self):
        """Composición indefinida incluye contexto físico para diagnóstico."""
        physics_fn = MagicMock(
            return_value={
                "success": True,
                "data": [1, 2, 3],
                "metrics": {},
            }
        )
        tactics_fn = MagicMock()

        result = compose_vectors(physics_fn, tactics_fn, ("f",), {})
        assert "physics_context" in result

    @pytest.mark.parametrize(
        "records",
        [
            [{"record_type": "A", "x": 1}],
            [
                {"record_type": "A", "x": 1},
                {"record_type": "A", "x": 2, "y": 3},
            ],
            [
                {"record_type": "A", "x": 1},
                {"record_type": "B", "y": 2},
            ],
        ],
    )
    def test_dimensionality_non_negative(self, records):
        """∀ tipo: dim ≥ 0."""
        dims = calculate_dimensionality(records)
        for dim in dims.values():
            assert dim >= 0

    def test_factory_reset_is_idempotent(self):
        """Resetear dos veces produce el mismo estado."""
        VectorFactory.reset_registry()
        r1 = set(VectorFactory._PHYSICS_REGISTRY.keys())

        VectorFactory.reset_registry()
        r2 = set(VectorFactory._PHYSICS_REGISTRY.keys())

        assert r1 == r2 == {"stabilize", "parse"}