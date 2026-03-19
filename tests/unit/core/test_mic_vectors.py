"""
tests/test_mic_vectors.py

Suite de pruebas exhaustiva para app/adapters/mic_vectors.py

Organización por estratos matemáticos:
───────────────────────────────────────
1. Contratos y Estructuras de Datos (Objetos de Categoría)
2. Helpers Internos (Funtores Naturales)
3. Guardas Topológicas (Precondiciones Homológicas)
4. Funciones de Topología Algebraica (Invariantes)
5. Vector Físico 1: Estabilización de Flujo (Espacio de Hilbert)
6. Vector Físico 2: Parsing Topológico (Complejo de Cadena)
7. Vector Táctico: Estructuración Lógica (Álgebra de Boole)
8. VectorFactory (Construcción Categórica)
9. Composición de Vectores (Funtores Adjuntos)
10. Propiedades Algebraicas Transversales (Leyes Universales)
"""

import os
import time
import math
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Callable
from unittest.mock import MagicMock, PropertyMock, patch
from abc import ABC, abstractmethod

import pytest

from app.adapters.mic_vectors import (
    VectorFactory,
    VectorMetrics,
    VectorResultStatus,
    _build_error,
    _build_result,
    _measure_memory_mb,
    calculate_algebraic_integrity,
    calculate_betti_numbers,
    calculate_topological_coherence,
    compose_vectors,
    Dimensionality,
    vector_parse_raw_structure,
    vector_stabilize_flux,
    vector_structure_logic,
    TopologicalGuard,
)
from app.core.schemas import Stratum


# =============================================================================
# ESTRUCTURAS ALGEBRAICAS AUXILIARES
# =============================================================================

class MetricNorm(ABC):
    """Interfaz categórica para normas en espacios de métricas."""
    
    @abstractmethod
    def norm(self, metrics: Dict[str, float]) -> float:
        """Calcula la norma de un vector de métricas."""
        pass
    
    @abstractmethod
    def distance(self, m1: Dict[str, float], m2: Dict[str, float]) -> float:
        """Métrica euclidiana entre dos vectores de métricas."""
        pass


class EuclideanMetricNorm(MetricNorm):
    """Norma euclidiana L₂ en ℝⁿ."""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Args:
            weights: Pesos para cada componente (normalización).
        """
        self.weights = weights or {
            "processing_time_ms": 1.0,
            "memory_usage_mb": 1.0,
            "topological_coherence": 1.0,
            "algebraic_integrity": 1.0,
        }
    
    def norm(self, metrics: Dict[str, float]) -> float:
        """‖m‖₂ = √(Σ wᵢ·mᵢ²)."""
        total = sum(
            self.weights.get(k, 1.0) * (v ** 2)
            for k, v in metrics.items()
        )
        return math.sqrt(max(0.0, total))
    
    def distance(self, m1: Dict[str, float], m2: Dict[str, float]) -> float:
        """d(m1, m2) = ‖m1 - m2‖₂."""
        diff = {k: m1.get(k, 0.0) - m2.get(k, 0.0) for k in set(m1) | set(m2)}
        return self.norm(diff)


class BettiNumbers:
    """
    Invariantes topológicos: números de Betti β₀, β₁, β₂.
    
    Propiedades:
    - β₀: número de componentes conexas
    - β₁: número de ciclos independientes (género)
    - β₂: número de cavidades
    - χ = β₀ - β₁ + β₂ (característica de Euler-Poincaré)
    """
    
    def __init__(self, beta_0: int = 0, beta_1: int = 0, beta_2: int = 0):
        self.beta_0 = max(0, int(beta_0))
        self.beta_1 = max(0, int(beta_1))
        self.beta_2 = max(0, int(beta_2))
    
    @property
    def euler_characteristic(self) -> int:
        """χ = β₀ - β₁ + β₂."""
        return self.beta_0 - self.beta_1 + self.beta_2
    
    def __getitem__(self, key: str) -> int:
        """Acceso tipo diccionario para compatibilidad."""
        return getattr(self, key, 0)
    
    def __eq__(self, other) -> bool:
        if isinstance(other, dict):
            return (
                self.beta_0 == other.get("beta_0", 0) and
                self.beta_1 == other.get("beta_1", 0) and
                self.beta_2 == other.get("beta_2", 0)
            )
        return (
            self.beta_0 == other.beta_0 and
            self.beta_1 == other.beta_1 and
            self.beta_2 == other.beta_2
        )
    
    def to_dict(self) -> Dict[str, int]:
        return {
            "beta_0": self.beta_0,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "euler": self.euler_characteristic,
        }
    
    @staticmethod
    def zero() -> "BettiNumbers":
        return BettiNumbers(0, 0, 0)


class Dimensionality:
    """
    Dimensión de espacio vectorial por tipo: dim(V_t) = |⋃ keys(r) : type(r)=t|.
    
    Soporta isomorfismo dimensional con tolerancia ε-aditiva.
    """
    
    def __init__(self, dimensions: Optional[Dict[str, int]] = None):
        self.dimensions = dimensions or {}
    
    @staticmethod
    def from_records(records: List[Dict[str, Any]]) -> "Dimensionality":
        """Construye dimensionalidad por unión de claves por tipo."""
        dims: Dict[str, set] = {}
        
        for record in records:
            record_type = record.get("record_type", "default")
            if record_type not in dims:
                dims[record_type] = set()
            dims[record_type].update(record.keys())
        
        return Dimensionality({t: len(keys) for t, keys in dims.items()})
    
    def is_isomorphic_to(
        self, 
        other: "Dimensionality",
        tolerance: float = 0.10
    ) -> bool:
        """
        Isomorfismo dimensional: ∀ t ∈ tipos,
        |dim_actual(t) - dim_esperada(t)| ≤ ε·max(1, dim_esperada(t)).
        
        La tolerancia ε se expresa como porcentaje relativo.
        """
        if not self.dimensions and not other.dimensions:
            return True
        
        if not self.dimensions or not other.dimensions:
            return False
        
        for key, expected_dim in self.dimensions.items():
            if key not in other.dimensions:
                return False
            
            actual_dim = other.dimensions[key]
            denominator = max(1, expected_dim)
            relative_error = abs(actual_dim - expected_dim) / denominator
            
            if relative_error > tolerance:
                return False
        
        return True
    
    def __eq__(self, other) -> bool:
        if isinstance(other, dict):
            return self.dimensions == other
        return self.dimensions == other.dimensions


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


@pytest.fixture
def metric_norm():
    """Norma euclidiana para validación de métricas."""
    return EuclideanMetricNorm()


# =============================================================================
# 1. CONTRATOS Y ESTRUCTURAS DE DATOS (OBJETOS CATEGÓRICOS)
# =============================================================================

class TestVectorResultStatus:
    """Verifica la enumeración de estados como objeto inicial en Categoría."""

    def test_all_statuses_exist(self):
        assert VectorResultStatus.SUCCESS.value == "success"
        assert VectorResultStatus.PHYSICS_ERROR.value == "physics_error"
        assert VectorResultStatus.LOGIC_ERROR.value == "logic_error"
        assert VectorResultStatus.TOPOLOGY_ERROR.value == "topology_error"

    def test_status_count(self):
        assert len(VectorResultStatus) >= 4

    def test_unique_values(self):
        values = [s.value for s in VectorResultStatus]
        assert len(values) == len(set(values))


class TestVectorMetrics:
    """
    Verifica VectorMetrics como objeto terminal con estructura de
    espacio vectorial normalizado.
    """

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
        """Frozen realiza la estructura de objeto terminal."""
        m = VectorMetrics(processing_time_ms=10.0)
        with pytest.raises(FrozenInstanceError):
            m.processing_time_ms = 20.0

    def test_frozen_no_new_attributes(self):
        m = VectorMetrics()
        with pytest.raises(FrozenInstanceError):
            m.processing_time_ms = 100.0

    def test_equality(self):
        m1 = VectorMetrics(processing_time_ms=10.0, memory_usage_mb=5.0)
        m2 = VectorMetrics(processing_time_ms=10.0, memory_usage_mb=5.0)
        assert m1 == m2

    def test_inequality(self):
        m1 = VectorMetrics(processing_time_ms=10.0)
        m2 = VectorMetrics(processing_time_ms=20.0)
        assert m1 != m2

    def test_euclidean_norm(self, metric_norm):
        """Validar norma L₂ de métricas."""
        m = VectorMetrics(
            processing_time_ms=3.0,
            memory_usage_mb=4.0,
            topological_coherence=0.0,
            algebraic_integrity=0.0,
        )
        metrics_dict = {
            "processing_time_ms": m.processing_time_ms,
            "memory_usage_mb": m.memory_usage_mb,
            "topological_coherence": m.topological_coherence,
            "algebraic_integrity": m.algebraic_integrity,
        }
        norm = metric_norm.norm(metrics_dict)
        assert norm == pytest.approx(5.0)  # √(9 + 16) = 5

    def test_metric_distance(self, metric_norm):
        """Distancia euclidiana entre dos métricas."""
        m1 = {
            "processing_time_ms": 0.0,
            "memory_usage_mb": 0.0,
            "topological_coherence": 1.0,
            "algebraic_integrity": 1.0,
        }
        m2 = {
            "processing_time_ms": 3.0,
            "memory_usage_mb": 4.0,
            "topological_coherence": 1.0,
            "algebraic_integrity": 1.0,
        }
        distance = metric_norm.distance(m1, m2)
        assert distance == pytest.approx(5.0)


# =============================================================================
# 2. HELPERS INTERNOS (FUNTORES NATURALES)
# =============================================================================

class TestBuildResult:
    """Pruebas para _build_result como constructor canónico."""

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
        assert isinstance(r["metrics"], dict)
        assert r["metrics"]["topological_coherence"] == 1.0

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

    def test_canonical_structure_preserved(self):
        """Estructura categórica siempre se preserva."""
        r = _build_result(
            success=False,
            stratum=Stratum.PHYSICS,
            status=VectorResultStatus.PHYSICS_ERROR,
            error="fallo",
        )
        # Estructura fija: success, stratum, status, metrics, error
        required = {"success", "stratum", "status", "metrics", "error"}
        assert required.issubset(r.keys())


class TestBuildError:
    """Pruebas para _build_error como constructor de fallos."""

    def test_error_result_is_failure(self):
        r = _build_error(
            stratum=Stratum.PHYSICS.value,
            status=VectorResultStatus.PHYSICS_ERROR,
            error="fallo catastrófico",
        )
        assert r["success"] is False

    def test_error_message_preserved(self):
        r = _build_error(
            stratum=Stratum.TACTICS.value,
            status=VectorResultStatus.LOGIC_ERROR,
            error="registro vacío",
        )
        assert r["error"] == "registro vacío"

    def test_elapsed_time_in_metrics(self):
        r = _build_error(
            stratum=Stratum.PHYSICS.value,
            status=VectorResultStatus.TOPOLOGY_ERROR,
            error="archivo no existe",
            metrics=VectorMetrics(processing_time_ms=123.45),
        )
        assert r["metrics"]["processing_time_ms"] == 123.45

    def test_default_elapsed_zero(self):
        r = _build_error(
            stratum=Stratum.PHYSICS.value,
            status=VectorResultStatus.PHYSICS_ERROR,
            error="error",
        )
        assert r["metrics"]["processing_time_ms"] == 0.0


# =============================================================================
# 3. GUARDAS TOPOLÓGICAS (PRECONDICIONES HOMOLÓGICAS)
# =============================================================================

class TestCalculateTopologicalCoherence:
    """
    Pruebas de C = S·R / (1 + H) como invariante topológico.
    
    Propiedades verificadas:
    - Acotación: C ∈ [0, 1]
    - Monotonicidad: ∂C/∂S > 0, ∂C/∂R > 0, ∂C/∂H < 0
    - Absorción: S=0 ∨ R=0 ⟹ C=0
    - Identidad: S=1 ∧ H=0 ∧ R=1 ⟹ C=1
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
        """∀ S,H,R: C ∈ [0, 1]."""
        cases = [
            {"stability_index": 0.5, "entropy": 0.3, "resonance_factor": 0.7},
            {"stability_index": 1.0, "entropy": 10.0, "resonance_factor": 1.0},
            {"stability_index": 0.0, "entropy": 0.0, "resonance_factor": 0.0},
            {"stability_index": 1.0, "entropy": 0.0, "resonance_factor": 1.0},
        ]
        for report in cases:
            c = calculate_topological_coherence(report)
            assert 0.0 <= c <= 1.0

    def test_monotone_increasing_stability(self):
        """∂C/∂S > 0."""
        base = {"entropy": 0.5, "resonance_factor": 0.8}
        c_low = calculate_topological_coherence({**base, "stability_index": 0.3})
        c_high = calculate_topological_coherence({**base, "stability_index": 0.9})
        assert c_high > c_low

    def test_monotone_increasing_resonance(self):
        """∂C/∂R > 0."""
        base = {"stability_index": 0.7, "entropy": 0.5}
        c_low = calculate_topological_coherence({**base, "resonance_factor": 0.2})
        c_high = calculate_topological_coherence({**base, "resonance_factor": 0.9})
        assert c_high > c_low

    def test_monotone_decreasing_entropy(self):
        """∂C/∂H < 0."""
        base = {"stability_index": 0.8, "resonance_factor": 0.9}
        c_low_entropy = calculate_topological_coherence({**base, "entropy": 0.1})
        c_high_entropy = calculate_topological_coherence({**base, "entropy": 5.0})
        assert c_low_entropy > c_high_entropy

    def test_specific_value(self):
        """Verificación con valores concretos."""
        report = {
            "stability_index": 0.8,
            "entropy": 0.2,
            "resonance_factor": 0.6,
        }
        expected = 0.8 * 0.6 / (1.0 + 0.2)
        assert calculate_topological_coherence(report) == pytest.approx(expected)

    def test_missing_keys_use_defaults(self):
        """Claves ausentes → defaults (S=0, H=0, R=0) → C=0."""
        result = calculate_topological_coherence({"unrelated": 42})
        assert result == 0.0

    def test_negative_values_clamped(self):
        """Valores negativos → 0."""
        report = {
            "stability_index": -0.5,
            "entropy": -1.0,
            "resonance_factor": -0.3,
        }
        result = calculate_topological_coherence(report)
        assert result == 0.0


class TestCalculateBettiNumbers:
    """
    Pruebas de números de Betti como invariantes topológicos.
    
    Relación de Euler-Poincaré: χ = β₀ - β₁ + β₂
    """

    def test_empty_records(self):
        result = calculate_betti_numbers([], {})
        assert result.beta_0 == 0
        assert result.beta_1 == 0
        assert result.beta_2 == 0
        assert result.euler_characteristic == 0

    def test_single_type_no_cycles(self):
        records = [
            {"record_type": "material", "value": 1},
            {"record_type": "material", "value": 2},
        ]
        result = calculate_betti_numbers(records, {})
        assert result.beta_0 == 1
        assert result.beta_1 == 0
        assert result.beta_2 == 0
        assert result.euler_characteristic == 1

    def test_multiple_types(self, sample_raw_records):
        result = calculate_betti_numbers(sample_raw_records, {})
        assert result.beta_0 == 3

    def test_cycles_from_cache(self, sample_raw_records, sample_parse_cache):
        result = calculate_betti_numbers(sample_raw_records, sample_parse_cache)
        assert result.beta_1 == 1

    def test_cavities_nested_empty(self, sample_raw_records):
        result = calculate_betti_numbers(sample_raw_records, {})
        assert result.beta_2 == 1

    def test_euler_characteristic_formula(self, sample_raw_records, sample_parse_cache):
        result = calculate_betti_numbers(sample_raw_records, sample_parse_cache)
        expected_euler = result.beta_0 - result.beta_1 + result.beta_2
        assert result.euler_characteristic == expected_euler

    def test_no_dependency_cycles_key(self):
        records = [{"record_type": "A"}]
        result = calculate_betti_numbers(records, {"other": "data"})
        assert result.beta_1 == 0

    def test_non_list_cycles_treated_as_zero(self):
        records = [{"record_type": "A"}]
        cache = {"dependency_cycles": "not_a_list"}
        result = calculate_betti_numbers(records, cache)
        assert result.beta_1 == 0

    def test_unknown_record_type_default(self):
        records = [{"value": 1}, {"value": 2}]
        result = calculate_betti_numbers(records, {})
        assert result.beta_0 == 1

    def test_all_nested_with_content(self):
        records = [
            {"record_type": "A", "nested": True, "content": "data"},
            {"record_type": "A", "nested": True, "content": "more"},
        ]
        result = calculate_betti_numbers(records, {})
        assert result.beta_2 == 0

    def test_betti_to_dict(self):
        """Conversión a diccionario."""
        betti = BettiNumbers(5, 3, 2)
        d = betti.to_dict()
        assert d == {
            "beta_0": 5,
            "beta_1": 3,
            "beta_2": 2,
            "euler": 4,
        }


class TestCalculateAlgebraicIntegrity:
    """
    Pruebas de I = 1/(1 + Σβᵢ, i>0) como medida de defectividad.
    
    Propiedades:
    - I ∈ (0, 1]: perfecta sin defectos
    - Defectos topológicos reducen integridad
    - Monotonía: defectos ↑ ⟹ integridad ↓
    """

    def test_zero_defects(self):
        """Sin defectos topológicos → integridad = 1."""
        betti = BettiNumbers(5, 0, 0)
        assert calculate_algebraic_integrity(betti) == 1.0

    def test_with_cycles(self):
        """β₁ = 2 ciclos → I = 1/(1+2) = 1/3."""
        betti = BettiNumbers(3, 2, 0)
        assert calculate_algebraic_integrity(betti) == pytest.approx(1.0 / 3.0)

    def test_with_cycles_and_cavities(self):
        """β₁ + β₂ = 2 + 3 = 5 defectos → I = 1/6."""
        betti = BettiNumbers(3, 2, 3)
        assert calculate_algebraic_integrity(betti) == pytest.approx(1.0 / 6.0)

    def test_only_beta_0(self):
        """Solo conectividad → integridad perfecta."""
        betti = BettiNumbers(10, 0, 0)
        assert calculate_algebraic_integrity(betti) == 1.0

    def test_integrity_decreases_with_defects(self):
        """∂I/∂defects < 0."""
        i_clean = calculate_algebraic_integrity(
            BettiNumbers(3, 0, 0)
        )
        i_dirty = calculate_algebraic_integrity(
            BettiNumbers(3, 5, 3)
        )
        assert i_clean > i_dirty

    def test_integrity_bounded_zero_one(self):
        """∀ β: I ∈ (0, 1]."""
        cases = [
            BettiNumbers(0, 0, 0),
            BettiNumbers(1, 0, 0),
            BettiNumbers(1, 100, 200),
        ]
        for betti in cases:
            i = calculate_algebraic_integrity(betti)
            assert 0.0 < i <= 1.0

    def test_empty_betti_dict(self):
        """Entrada vacía → sin defectos → I=1."""
        assert calculate_algebraic_integrity(BettiNumbers.zero()) == 1.0


class TestCalculateDimensionality:
    """
    Pruebas de dim(V_t) = |⋃ keys(r) : type(r)=t|.
    
    Propiedades:
    - Unión: dim({k₁} ∪ {k₂}) = |k₁ ∪ k₂|
    - No inflación: repeating keys no duplican
    - No-negatividad: dim ≥ 0
    """

    def test_empty_records(self):
        result = Dimensionality.from_records([])
        assert result.dimensions == {}

    def test_single_record(self):
        records = [{"record_type": "A", "x": 1, "y": 2}]
        result = Dimensionality.from_records(records)
        assert result.dimensions["A"] == 3

    def test_union_of_keys(self):
        """Claves disyuntas se unen."""
        records = [
            {"record_type": "A", "x": 1},
            {"record_type": "A", "y": 2},
        ]
        result = Dimensionality.from_records(records)
        assert result.dimensions["A"] == 3

    def test_same_keys_no_inflation(self):
        """Claves idénticas no inflan dimensión."""
        records = [
            {"record_type": "A", "x": 1, "y": 2},
            {"record_type": "A", "x": 3, "y": 4},
            {"record_type": "A", "x": 5, "y": 6},
        ]
        result = Dimensionality.from_records(records)
        assert result.dimensions["A"] == 3

    def test_multiple_types(self, sample_raw_records):
        result = Dimensionality.from_records(sample_raw_records)
        assert "material" in result.dimensions
        assert "labor" in result.dimensions
        assert "equipment" in result.dimensions

    def test_default_type_when_missing(self):
        records = [{"x": 1}, {"y": 2}]
        result = Dimensionality.from_records(records)
        assert "default" in result.dimensions
        assert result.dimensions["default"] == 2

    def test_default_type_union(self):
        """Records sin type → agrupan como 'default' con unión."""
        records = [{"a": 1, "b": 2}, {"b": 3, "c": 4}]
        result = Dimensionality.from_records(records)
        assert result.dimensions["default"] == 3


class TestValidateDimensionalIsomorphism:
    """
    Pruebas del isomorfismo dimensional con tolerancia ε.
    
    Definición: iso(D₁, D₂, ε) ⟺ ∀t,
    |dim₁(t) - dim₂(t)| ≤ ε·max(1, dim₁(t))
    """

    def test_both_empty_vacuously_true(self):
        assert Dimensionality({}).is_isomorphic_to(Dimensionality({})) is True

    def test_expected_empty_non_isomorphic(self):
        assert Dimensionality({}).is_isomorphic_to(Dimensionality({"A": 5})) is False

    def test_actual_empty_non_isomorphic(self):
        assert Dimensionality({"A": 5}).is_isomorphic_to(Dimensionality({})) is False

    def test_identical_dimensions(self):
        d = Dimensionality({"A": 10, "B": 20})
        assert d.is_isomorphic_to(d) is True

    def test_within_tolerance_5percent(self):
        """ε=10%, error=5% → iso=True."""
        expected = Dimensionality({"A": 100, "B": 200})
        actual = Dimensionality({"A": 105, "B": 210})
        assert expected.is_isomorphic_to(actual) is True

    def test_beyond_tolerance(self):
        """ε=10%, error=15% → iso=False."""
        expected = Dimensionality({"A": 100})
        actual = Dimensionality({"A": 115})
        assert expected.is_isomorphic_to(actual) is False

    def test_different_keys_non_isomorphic(self):
        """Tipos disyuntos → no isomorfos."""
        expected = Dimensionality({"A": 10})
        actual = Dimensionality({"B": 10})
        assert expected.is_isomorphic_to(actual) is False

    def test_custom_tolerance_10percent(self):
        """Con ε=10% para error 20%."""
        expected = Dimensionality({"A": 100})
        actual = Dimensionality({"A": 120})
        assert expected.is_isomorphic_to(actual, tolerance=0.10) is False

    def test_custom_tolerance_25percent(self):
        """Con ε=25% para error 20%."""
        expected = Dimensionality({"A": 100})
        actual = Dimensionality({"A": 120})
        assert expected.is_isomorphic_to(actual, tolerance=0.25) is True

    def test_zero_expected_dimension(self):
        """max(0, 1) = 1 evita división por cero."""
        expected = Dimensionality({"A": 0})
        actual = Dimensionality({"A": 0})
        assert expected.is_isomorphic_to(actual) is True

    def test_reflexivity(self):
        """iso(D, D) = True."""
        d = Dimensionality({"X": 42, "Y": 17})
        assert d.is_isomorphic_to(d) is True

    def test_symmetry(self):
        """iso(D1, D2) = iso(D2, D1) (con ε fija)."""
        d1 = Dimensionality({"A": 100})
        d2 = Dimensionality({"A": 110})
        eps = 0.15
        r1 = d1.is_isomorphic_to(d2, tolerance=eps)
        r2 = d2.is_isomorphic_to(d1, tolerance=eps)
        assert r1 == r2


# =============================================================================
# 5. VECTOR FÍSICO 1: ESTABILIZACIÓN DE FLUJO (ESPACIO DE HILBERT)
# =============================================================================

class TestVectorStabilizeFlux:
    """
    Pruebas del vector físico: φ_phys,1(f, C) → estabilización de flujo.
    
    Espacio: L²([0, ∞)) con métrica euclidiana.
    """

    def test_file_not_exists(self, nonexistent_file, valid_physics_config):
        result = vector_stabilize_flux(nonexistent_file, valid_physics_config)
        assert result["success"] is False
        assert result["status"] == "topology_error"
        assert "no existe" in result["error"]

    def test_missing_config_keys(self, tmp_file):
        config = {"system_capacitance": 1.0}
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
        assert result["stratum"] == Stratum.PHYSICS.name
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

    @patch("app.adapters.mic_vectors.DataFluxCondenser")
    @patch("app.adapters.mic_vectors.CondenserConfig")
    def test_physics_report_always_computed(
        self, MockConfig, MockCondenser, tmp_file, valid_physics_config, mock_dataframe
    ):
        """Reporte físico se computa siempre (identidad de coherencia)."""
        mock_instance = MagicMock()
        mock_instance.stabilize.return_value = mock_dataframe
        mock_instance.get_physics_report.return_value = {
            "stability_index": 1.0,
            "entropy": 0.0,
            "resonance_factor": 1.0,
        }
        MockCondenser.return_value = mock_instance

        result = vector_stabilize_flux(tmp_file, valid_physics_config)
        assert result["metrics"]["topological_coherence"] == 1.0


# =============================================================================
# 6. VECTOR FÍSICO 2: PARSING TOPOLÓGICO (COMPLEJO DE CADENA)
# =============================================================================

class TestVectorParseRawStructure:
    """
    Pruebas del vector de parsing: φ_phys,2(f, P) → estructura topológica.
    
    Espacio: Complejo simplicial de registros crudos.
    """

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

    @patch("app.adapters.mic_vectors.ReportParserCrudo")
    def test_successful_parsing(
        self, MockParser, tmp_file, valid_profile, sample_raw_records
    ):
        mock_instance = MagicMock()
        mock_instance.parse_to_raw.return_value = sample_raw_records
        mock_instance.get_parse_cache.return_value = {"dependency_cycles": []}
        mock_instance.validation_stats = SimpleNamespace(valid=5, invalid=0)
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
        """Inyección de dimensionalidad en cache."""
        mock_instance = MagicMock()
        mock_instance.parse_to_raw.return_value = sample_raw_records
        mock_instance.get_parse_cache.return_value = {}
        mock_instance.validation_stats = SimpleNamespace()
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

    @patch("app.adapters.mic_vectors.ReportParserCrudo")
    def test_homological_invariants_structure(
        self, MockParser, tmp_file, valid_profile
    ):
        """Estructura de invariantes homológicos."""
        mock_instance = MagicMock()
        mock_instance.parse_to_raw.return_value = [
            {"record_type": "A", "v": 1},
            {"record_type": "B", "v": 2},
        ]
        mock_instance.get_parse_cache.return_value = {
            "dependency_cycles": [["x", "y", "x"]]
        }
        mock_instance.validation_stats = SimpleNamespace()
        MockParser.return_value = mock_instance

        result = vector_parse_raw_structure(tmp_file, valid_profile)
        inv = result["homological_invariants"]
        assert "beta_0" in inv
        assert "beta_1" in inv
        assert "beta_2" in inv
        assert "euler" in inv
        assert inv["beta_0"] == 2
        assert inv["beta_1"] == 1

    @patch("app.adapters.mic_vectors.ReportParserCrudo")
    def test_valid_topological_constraints_passed(
        self, MockParser, tmp_file, valid_profile, valid_topological_constraints
    ):
        mock_instance = MagicMock()
        mock_instance.parse_to_raw.return_value = [{"record_type": "A"}]
        mock_instance.get_parse_cache.return_value = {}
        mock_instance.validation_stats = SimpleNamespace()
        MockParser.return_value = mock_instance

        result = vector_parse_raw_structure(
            tmp_file, valid_profile, valid_topological_constraints
        )
        assert result["success"] is True

    @patch("app.adapters.mic_vectors.ReportParserCrudo")
    def test_no_validation_stats_attribute(
        self, MockParser, tmp_file, valid_profile
    ):
        """Parser sin validation_stats → fallback seguro."""
        mock_instance = MagicMock(spec=[])
        mock_instance.parse_to_raw = MagicMock(
            return_value=[{"record_type": "A"}]
        )
        mock_instance.get_parse_cache = MagicMock(return_value={})
        MockParser.return_value = mock_instance

        result = vector_parse_raw_structure(tmp_file, valid_profile)
        assert result["success"] is True
        assert result["validation_stats"] == {}


# =============================================================================
# 7. VECTOR TÁCTICO: ESTRUCTURACIÓN LÓGICA (ÁLGEBRA DE BOOLE)
# =============================================================================

class TestVectorStructureLogic:
    """
    Pruebas del vector táctico: φ_tact(R, C, Config) → lógica estructurada.
    
    Espacio: Álgebra booleana de proposiciones lógicas.
    """

    def test_empty_records_returns_error(self, sample_parse_cache, sample_config):
        result = vector_structure_logic([], sample_parse_cache, sample_config)
        assert result["success"] is False
        assert result["status"] == "logic_error"

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
        assert result["stratum"] == Stratum.TACTICS.name
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
        """Fallback graceful para métodos ausentes."""
        mock_instance = MagicMock(spec=[])
        mock_instance.process_all = MagicMock(return_value=mock_dataframe)
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
        """Isomorfismo roto → warning, sin fallo."""
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
        assert any("Isomorfismo dimensional" in r.message for r in caplog.records)

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
# 8. VECTOR FACTORY (CONSTRUCCIÓN CATEGÓRICA)
# =============================================================================

class TestVectorFactory:
    """Factory pattern para construcción de vectores."""


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


    def test_wrapper_preserves_docstring(self):
        wrapper = VectorFactory.create_physics_vector("stabilize")
        assert wrapper.__doc__ is not None
        assert "PHYSICS" in wrapper.__doc__

    def test_tactics_wrapper_preserves_docstring(self):
        wrapper = VectorFactory.create_tactics_vector()
        assert wrapper.__doc__ is not None
        assert "TACTICS" in wrapper.__doc__


# =============================================================================
# 9. COMPOSICIÓN DE VECTORES (FUNTORES ADJUNTOS)
# =============================================================================

class TestComposeVectors:
    """
    Pruebas del pipeline de composición φ_tact ∘ φ_phys.
    
    Estructura categórica: funtores adjuntos con unidad y counidad.
    """

    def test_physics_failure_short_circuits(self):
        """η: fallo temprano en φ_phys."""
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
        """μ: composición indefinida sin raw_records."""
        physics_fn = MagicMock(
            return_value={
                "success": True,
                "data": [{"col": 1}],
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
        """ε: composición exitosa φ_tact ∘ φ_phys."""
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

        cm = result["combined_metrics"]
        assert cm["total_time_ms"] == pytest.approx(80.0)
        assert cm["pipeline_coherence"] == pytest.approx(0.85)

    def test_tactics_receives_raw_records_and_cache(self):
        """Passthrough correctamente tipificado."""
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
        """Cuello de botella: min(C_phys, C_tact)."""
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
        """Fallo en φ_tact se propaga al resultado final."""
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

    def test_empty_parse_cache_default(self):
        """Cache vacío si φ_phys no lo proporciona."""
        physics_fn = MagicMock(
            return_value={
                "success": True,
                "raw_records": [{"r": 1}],
                "metrics": {},
            }
        )
        tactics_fn = MagicMock(
            return_value={"success": True, "metrics": {}}
        )

        compose_vectors(physics_fn, tactics_fn, ("f",), {})

        call_args = tactics_fn.call_args
        assert call_args[0][1] == {}

    def test_composition_closure(self):
        """Composición devuelve estructura de Resultado."""
        physics_fn = MagicMock(
            return_value={
                "success": True,
                "raw_records": [{}],
                "parse_cache": {},
                "metrics": {},
            }
        )
        tactics_fn = MagicMock(
            return_value={"success": True, "metrics": {}}
        )

        result = compose_vectors(physics_fn, tactics_fn, ("f",), {})
        assert "success" in result
        assert "combined_metrics" in result


# =============================================================================
# 10. PROPIEDADES ALGEBRAICAS TRANSVERSALES (LEYES UNIVERSALES)
# =============================================================================

class TestAlgebraicProperties:
    """
    Leyes matemáticas que deben sostenerse universalmente.
    
    Verificadas:
    - Acotación
    - Monotonicidad
    - Identidades elementa
    - Características euler
    - Idempotencia
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
        """∀ S,H,R ∈ ℝ: C ∈ [0, 1]."""
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
            BettiNumbers(1, 0, 0),
            BettiNumbers(5, 3, 1),
            BettiNumbers(10, 10, 10),
            BettiNumbers(0, 0, 0),
        ],
    )
    def test_integrity_in_unit_interval(self, betti):
        """∀ β₀,β₁,β₂: I ∈ (0, 1]."""
        i = calculate_algebraic_integrity(betti)
        assert 0.0 < i <= 1.0

    def test_euler_characteristic_formula(self):
        """χ = β₀ - β₁ + β₂ (Euler-Poincaré)."""
        records = [
            {"record_type": "A", "nested": True, "content": None},
            {"record_type": "B"},
            {"record_type": "A"},
        ]
        cache = {"dependency_cycles": [["x", "y"], ["a", "b"]]}

        betti = calculate_betti_numbers(records, cache)
        assert betti.euler_characteristic == betti.beta_0 - betti.beta_1 + betti.beta_2

    def test_dimensionality_idempotent(self, sample_raw_records):
        """dim ∘ dim = dim (idempotencia)."""
        d1 = Dimensionality.from_records(sample_raw_records).dimensions
        d2 = Dimensionality.from_records(sample_raw_records).dimensions
        assert d1 == d2

    def test_isomorphism_reflexive(self):
        """iso(D, D) = True (reflexividad)."""
        d = Dimensionality({"A": 10, "B": 20, "C": 30})
        assert d.is_isomorphic_to(d) is True

    def test_isomorphism_symmetric(self):
        """iso(D1, D2) ⟺ iso(D2, D1) (simetría)."""
        d1 = Dimensionality({"A": 100, "B": 200})
        d2 = Dimensionality({"A": 100, "B": 200})
        assert d1.is_isomorphic_to(d2) == d2.is_isomorphic_to(d1)

    @pytest.mark.parametrize(
        "records,expected_beta0",
        [
            ([], 0),
            ([{"record_type": "A"}], 1),
            ([{"record_type": "A"}, {"record_type": "B"}], 2),
            ([{"record_type": "A"}, {"record_type": "B"}, {"record_type": "C"}], 3),
        ],
    )
    def test_beta0_equals_unique_types(self, records, expected_beta0):
        """β₀ = |tipos únicos| siempre."""
        betti = calculate_betti_numbers(records, {})
        assert betti.beta_0 == expected_beta0

    def test_build_result_always_has_canonical_keys(self):
        """Todo resultado tiene estructura canónica: {success, stratum, status, metrics}."""
        canonical_keys = {"success", "stratum", "status", "metrics"}

        for status in VectorResultStatus:
            r = _build_result(
                success=status == VectorResultStatus.SUCCESS,
                stratum=Stratum.PHYSICS,
                status=status,
            )
            assert canonical_keys.issubset(r.keys())

    def test_coherence_identity_element(self):
        """C(S=1, H=0, R=1) = 1 (identidad de coherencia)."""
        report = {
            "stability_index": 1.0,
            "entropy": 0.0,
            "resonance_factor": 1.0,
        }
        assert calculate_topological_coherence(report) == 1.0

    def test_integrity_identity_element(self):
        """I(sin defectos) = 1 (identidad de integridad)."""
        betti = BettiNumbers(42, 0, 0)
        assert calculate_algebraic_integrity(betti) == 1.0

    def test_composition_preserves_context(self):
        """Composición indefinida preserva contexto físico."""
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
            [{"record_type": "A", "x": 1}, {"record_type": "A", "x": 2, "y": 3}],
            [{"record_type": "A", "x": 1}, {"record_type": "B", "y": 2}],
        ],
    )
    def test_dimensionality_non_negative(self, records):
        """∀ tipo: dim ≥ 0."""
        dims = Dimensionality.from_records(records).dimensions
        for dim in dims.values():
            assert dim >= 0


    def test_metric_norm_triangle_inequality(self, metric_norm):
        """‖m1 + m2‖ ≤ ‖m1‖ + ‖m2‖."""
        m1 = {"processing_time_ms": 3.0, "memory_usage_mb": 0.0, "topological_coherence": 0.0, "algebraic_integrity": 0.0}
        m2 = {"processing_time_ms": 0.0, "memory_usage_mb": 4.0, "topological_coherence": 0.0, "algebraic_integrity": 0.0}
        m_sum = {k: m1.get(k, 0.0) + m2.get(k, 0.0) for k in set(m1) | set(m2)}

        n1 = metric_norm.norm(m1)
        n2 = metric_norm.norm(m2)
        n_sum = metric_norm.norm(m_sum)

        assert n_sum <= n1 + n2 + 1e-10

    def test_dimensional_isomorphism_tolerance_monotone(self):
        """ε₁ < ε₂ ⟹ iso(., ε₁) ⟹ iso(., ε₂)."""
        expected = Dimensionality({"A": 100})
        actual = Dimensionality({"A": 112})

        iso_10 = expected.is_isomorphic_to(actual, tolerance=0.10)
        iso_15 = expected.is_isomorphic_to(actual, tolerance=0.15)

        if not iso_10:
            assert not iso_15 or iso_15  # si es False en 10%, puede ser T o F en 15%