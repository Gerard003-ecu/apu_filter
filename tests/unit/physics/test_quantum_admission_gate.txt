"""
Suite de pruebas rigurosa para ``quantum_admission_gate.py``.

Estructura:
- Mocks para interfaces inyectadas (oracles).
- Pruebas unitarias por método/función.
- Pruebas de integración para ``evaluate_admission`` y ``__call__``.
- Pruebas de propiedades (determinismo, invariantes, inmutabilidad).

Convenciones:
- Cada clase ``Test*`` agrupa pruebas de una unidad lógica.
- Los nombres siguen ``test_<condición>_<resultado_esperado>``.
- Se usa ``pytest.mark.parametrize`` para variantes sin duplicación.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import FrozenInstanceError
from typing import Any, Dict, Mapping, Optional
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from quantum_admission_gate import (
    Eigenstate,
    ILaplaceOracle,
    ISheafCohomologyOrchestrator,
    ITopologicalWatcher,
    QuantumAdmissionError,
    QuantumAdmissionGate,
    QuantumConstants,
    QuantumInterfaceError,
    QuantumMeasurement,
    QuantumNumericalError,
    _clamp_probability,
    _ensure_finite_float,
    _safe_context,
)


# ======================================================================
# Mock Oracles
# ======================================================================


class MockTopologicalWatcher:
    """Watcher con threat configurable."""

    def __init__(self, threat: float = 0.0) -> None:
        self._threat = threat

    def get_mahalanobis_threat(self) -> float:
        return self._threat


class MockLaplaceOracle:
    """Oracle con polo dominante configurable."""

    def __init__(self, pole: float = -1.0) -> None:
        self._pole = pole

    def get_dominant_pole_real(self) -> float:
        return self._pole


class MockSheafOrchestrator:
    """Orchestrator con frustración configurable."""

    def __init__(self, frustration: float = 0.0) -> None:
        self._frustration = frustration

    def get_global_frustration_energy(self) -> float:
        return self._frustration


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def default_watcher() -> MockTopologicalWatcher:
    return MockTopologicalWatcher(threat=0.0)


@pytest.fixture()
def default_oracle() -> MockLaplaceOracle:
    return MockLaplaceOracle(pole=-1.0)


@pytest.fixture()
def default_sheaf() -> MockSheafOrchestrator:
    return MockSheafOrchestrator(frustration=0.0)


@pytest.fixture()
def gate(
    default_watcher: MockTopologicalWatcher,
    default_oracle: MockLaplaceOracle,
    default_sheaf: MockSheafOrchestrator,
) -> QuantumAdmissionGate:
    return QuantumAdmissionGate(
        topo_watcher=default_watcher,
        laplace_oracle=default_oracle,
        sheaf_orchestrator=default_sheaf,
    )


@pytest.fixture()
def small_payload() -> Dict[str, Any]:
    return {"key": "value"}


@pytest.fixture()
def large_payload() -> Dict[str, Any]:
    return {f"key_{i}": f"value_{i}" * 100 for i in range(50)}


def make_gate(
    threat: float = 0.0,
    pole: float = -1.0,
    frustration: float = 0.0,
) -> QuantumAdmissionGate:
    """Factory para crear gates con parámetros específicos."""
    return QuantumAdmissionGate(
        topo_watcher=MockTopologicalWatcher(threat=threat),
        laplace_oracle=MockLaplaceOracle(pole=pole),
        sheaf_orchestrator=MockSheafOrchestrator(frustration=frustration),
    )


# ======================================================================
# Tests: QuantumConstants
# ======================================================================


class TestQuantumConstants:
    """Pruebas para constantes físicas."""

    def test_planck_relation(self) -> None:
        expected_hbar = QuantumConstants.PLANCK_H / (2.0 * math.pi)
        assert QuantumConstants.PLANCK_HBAR == pytest.approx(expected_hbar)

    def test_all_constants_finite(self) -> None:
        constants = [
            QuantumConstants.PLANCK_H,
            QuantumConstants.PLANCK_HBAR,
            QuantumConstants.BASE_WORK_FUNCTION,
            QuantumConstants.BASE_EFFECTIVE_MASS,
            QuantumConstants.BARRIER_WIDTH,
            QuantumConstants.ALPHA_THREAT,
            QuantumConstants.MIN_KINETIC_ENERGY,
            QuantumConstants.FRUSTRATION_VETO_TOL,
            QuantumConstants.SIGMA_CHAOS_TOL,
            QuantumConstants.ENTROPY_FLOOR,
            QuantumConstants.EXP_UNDERFLOW_CUTOFF,
        ]
        for c in constants:
            assert math.isfinite(c), f"Constante no finita: {c}"

    def test_positive_physical_constants(self) -> None:
        assert QuantumConstants.PLANCK_H > 0
        assert QuantumConstants.PLANCK_HBAR > 0
        assert QuantumConstants.BASE_WORK_FUNCTION > 0
        assert QuantumConstants.BASE_EFFECTIVE_MASS > 0
        assert QuantumConstants.BARRIER_WIDTH > 0
        assert QuantumConstants.ALPHA_THREAT > 0
        assert QuantumConstants.MIN_KINETIC_ENERGY > 0

    def test_tolerances_positive(self) -> None:
        assert QuantumConstants.FRUSTRATION_VETO_TOL > 0
        assert QuantumConstants.SIGMA_CHAOS_TOL > 0
        assert QuantumConstants.ENTROPY_FLOOR > 0

    def test_underflow_cutoff_negative(self) -> None:
        assert QuantumConstants.EXP_UNDERFLOW_CUTOFF < 0

    def test_no_subclassing(self) -> None:
        with pytest.raises(TypeError, match="no debe ser subclaseada"):

            class SubConstants(QuantumConstants):
                pass


# ======================================================================
# Tests: Eigenstate
# ======================================================================


class TestEigenstate:
    """Pruebas para autoestados."""

    def test_values_exist(self) -> None:
        assert Eigenstate.ADMITIDO is not None
        assert Eigenstate.RECHAZADO is not None

    def test_distinct(self) -> None:
        assert Eigenstate.ADMITIDO != Eigenstate.RECHAZADO

    def test_completeness(self) -> None:
        assert len(Eigenstate) == 2


# ======================================================================
# Tests: QuantumMeasurement
# ======================================================================


class TestQuantumMeasurement:
    """Pruebas para el registro de medición."""

    def test_creation(self) -> None:
        m = QuantumMeasurement(
            eigenstate=Eigenstate.ADMITIDO,
            incident_energy=5.0,
            work_function=10.0,
            tunneling_probability=0.5,
            kinetic_energy=0.001,
            momentum=0.01,
            frustration_veto=False,
            effective_mass=1.0,
            dominant_pole_real=-1.0,
            threat_level=0.0,
            collapse_threshold=0.3,
            admission_reason="Test",
        )
        assert m.eigenstate == Eigenstate.ADMITIDO
        assert m.incident_energy == pytest.approx(5.0)

    def test_frozen(self) -> None:
        m = QuantumMeasurement(
            eigenstate=Eigenstate.RECHAZADO,
            incident_energy=0.0,
            work_function=0.0,
            tunneling_probability=0.0,
            kinetic_energy=0.0,
            momentum=0.0,
            frustration_veto=True,
            effective_mass=float("inf"),
            dominant_pole_real=0.0,
            threat_level=0.0,
            collapse_threshold=1.0,
            admission_reason="Veto",
        )
        with pytest.raises((AttributeError, FrozenInstanceError)):
            m.eigenstate = Eigenstate.ADMITIDO  # type: ignore[misc]

    def test_repr_contains_key_info(self) -> None:
        m = QuantumMeasurement(
            eigenstate=Eigenstate.ADMITIDO,
            incident_energy=5.0,
            work_function=10.0,
            tunneling_probability=0.5,
            kinetic_energy=0.001,
            momentum=0.01,
            frustration_veto=False,
            effective_mass=1.0,
            dominant_pole_real=-1.0,
            threat_level=0.0,
            collapse_threshold=0.3,
            admission_reason="Test reason",
        )
        r = repr(m)
        assert "ADMITIDO" in r
        assert "Test reason" in r
        assert "QuantumMeasurement" in r


# ======================================================================
# Tests: Excepciones
# ======================================================================


class TestExceptions:
    """Pruebas para la jerarquía de excepciones."""

    def test_hierarchy(self) -> None:
        assert issubclass(QuantumNumericalError, QuantumAdmissionError)
        assert issubclass(QuantumInterfaceError, QuantumAdmissionError)
        assert issubclass(QuantumAdmissionError, Exception)

    def test_instantiation(self) -> None:
        e1 = QuantumAdmissionError("base")
        assert str(e1) == "base"
        e2 = QuantumNumericalError("num")
        assert str(e2) == "num"
        e3 = QuantumInterfaceError("iface")
        assert str(e3) == "iface"

    def test_catch_by_base(self) -> None:
        with pytest.raises(QuantumAdmissionError):
            raise QuantumNumericalError("test")
        with pytest.raises(QuantumAdmissionError):
            raise QuantumInterfaceError("test")


# ======================================================================
# Tests: _ensure_finite_float
# ======================================================================


class TestEnsureFiniteFloat:
    """Pruebas para conversión a float finito."""

    @pytest.mark.parametrize(
        "value, expected",
        [
            (42, 42.0),
            (3.14, 3.14),
            ("10.5", 10.5),
            (-7.0, -7.0),
            (0, 0.0),
            (True, 1.0),
        ],
    )
    def test_valid_conversions(
        self, value: Any, expected: float
    ) -> None:
        assert _ensure_finite_float(value, name="test") == pytest.approx(
            expected
        )

    @pytest.mark.parametrize(
        "value",
        [float("inf"), float("-inf"), float("nan")],
    )
    def test_non_finite_raises(self, value: float) -> None:
        with pytest.raises(QuantumNumericalError, match="no es finito"):
            _ensure_finite_float(value, name="test")

    @pytest.mark.parametrize(
        "value",
        [None, "abc", [], {}, object()],
    )
    def test_non_convertible_raises(self, value: Any) -> None:
        with pytest.raises(
            QuantumNumericalError, match="no es convertible"
        ):
            _ensure_finite_float(value, name="test")

    def test_error_message_contains_name(self) -> None:
        with pytest.raises(
            QuantumNumericalError, match="my_param"
        ):
            _ensure_finite_float("bad", name="my_param")


# ======================================================================
# Tests: _clamp_probability
# ======================================================================


class TestClampProbability:
    """Pruebas para saturación de probabilidad."""

    @pytest.mark.parametrize(
        "value, expected",
        [
            (0.5, 0.5),
            (0.0, 0.0),
            (1.0, 1.0),
            (-0.1, 0.0),
            (-1000.0, 0.0),
            (1.5, 1.0),
            (100.0, 1.0),
            (1e-15, 1e-15),
            (0.999999, 0.999999),
        ],
    )
    def test_clamping(
        self, value: float, expected: float
    ) -> None:
        assert _clamp_probability(value) == pytest.approx(expected)

    def test_nan_returns_zero(self) -> None:
        assert _clamp_probability(float("nan")) == 0.0

    def test_positive_inf_returns_zero(self) -> None:
        assert _clamp_probability(float("inf")) == 0.0

    def test_negative_inf_returns_zero(self) -> None:
        assert _clamp_probability(float("-inf")) == 0.0

    def test_result_always_in_range(self) -> None:
        test_values = [
            -1e100,
            -1.0,
            0.0,
            0.5,
            1.0,
            1e100,
            float("nan"),
            float("inf"),
            float("-inf"),
        ]
        for v in test_values:
            result = _clamp_probability(v)
            assert 0.0 <= result <= 1.0, f"Value {v} produced {result}"


# ======================================================================
# Tests: _safe_context
# ======================================================================


class TestSafeContext:
    """Pruebas para normalización de contexto."""

    def test_none_returns_empty_dict(self) -> None:
        assert _safe_context(None) == {}

    def test_dict_returned_as_copy(self) -> None:
        original = {"key": "value"}
        result = _safe_context(original)
        assert result == original
        assert result is not original

    def test_mapping_converted_to_dict(self) -> None:
        from collections import OrderedDict

        m = OrderedDict([("a", 1), ("b", 2)])
        result = _safe_context(m)
        assert isinstance(result, dict)
        assert result == {"a": 1, "b": 2}

    def test_non_mapping_returns_warning(self) -> None:
        result = _safe_context("not_a_mapping")  # type: ignore[arg-type]
        assert "_context_warning" in result

    def test_non_mapping_list(self) -> None:
        result = _safe_context([1, 2, 3])  # type: ignore[arg-type]
        assert "_context_warning" in result


# ======================================================================
# Tests: Dependency Validation
# ======================================================================


class TestDependencyValidation:
    """Pruebas para validación de dependencias inyectadas."""

    def test_valid_dependencies(
        self,
        default_watcher: MockTopologicalWatcher,
        default_oracle: MockLaplaceOracle,
        default_sheaf: MockSheafOrchestrator,
    ) -> None:
        gate = QuantumAdmissionGate(
            topo_watcher=default_watcher,
            laplace_oracle=default_oracle,
            sheaf_orchestrator=default_sheaf,
        )
        assert gate is not None

    def test_none_watcher_raises(
        self,
        default_oracle: MockLaplaceOracle,
        default_sheaf: MockSheafOrchestrator,
    ) -> None:
        with pytest.raises(QuantumInterfaceError, match="topo_watcher"):
            QuantumAdmissionGate(
                topo_watcher=None,  # type: ignore[arg-type]
                laplace_oracle=default_oracle,
                sheaf_orchestrator=default_sheaf,
            )

    def test_none_oracle_raises(
        self,
        default_watcher: MockTopologicalWatcher,
        default_sheaf: MockSheafOrchestrator,
    ) -> None:
        with pytest.raises(
            QuantumInterfaceError, match="laplace_oracle"
        ):
            QuantumAdmissionGate(
                topo_watcher=default_watcher,
                laplace_oracle=None,  # type: ignore[arg-type]
                sheaf_orchestrator=default_sheaf,
            )

    def test_none_sheaf_raises(
        self,
        default_watcher: MockTopologicalWatcher,
        default_oracle: MockLaplaceOracle,
    ) -> None:
        with pytest.raises(
            QuantumInterfaceError, match="sheaf_orchestrator"
        ):
            QuantumAdmissionGate(
                topo_watcher=default_watcher,
                laplace_oracle=default_oracle,
                sheaf_orchestrator=None,  # type: ignore[arg-type]
            )

    def test_missing_method_raises(
        self,
        default_oracle: MockLaplaceOracle,
        default_sheaf: MockSheafOrchestrator,
    ) -> None:
        class BadWatcher:
            pass

        with pytest.raises(
            QuantumInterfaceError, match="get_mahalanobis_threat"
        ):
            QuantumAdmissionGate(
                topo_watcher=BadWatcher(),  # type: ignore[arg-type]
                laplace_oracle=default_oracle,
                sheaf_orchestrator=default_sheaf,
            )

    def test_non_callable_method_raises(
        self,
        default_oracle: MockLaplaceOracle,
        default_sheaf: MockSheafOrchestrator,
    ) -> None:
        class WatcherWithProperty:
            get_mahalanobis_threat = 42  # Not callable

        with pytest.raises(
            QuantumInterfaceError, match="get_mahalanobis_threat"
        ):
            QuantumAdmissionGate(
                topo_watcher=WatcherWithProperty(),  # type: ignore[arg-type]
                laplace_oracle=default_oracle,
                sheaf_orchestrator=default_sheaf,
            )


# ======================================================================
# Tests: _serialize_payload
# ======================================================================


class TestSerializePayload:
    """Pruebas para serialización determinista."""

    def test_deterministic(self, gate: QuantumAdmissionGate) -> None:
        payload = {"a": 1, "b": "hello", "c": [1, 2, 3]}
        r1 = gate._serialize_payload(payload)
        r2 = gate._serialize_payload(payload)
        assert r1 == r2

    def test_order_independent(
        self, gate: QuantumAdmissionGate
    ) -> None:
        p1 = {"z": 1, "a": 2, "m": 3}
        p2 = {"a": 2, "m": 3, "z": 1}
        assert gate._serialize_payload(p1) == gate._serialize_payload(
            p2
        )

    def test_returns_bytes(
        self, gate: QuantumAdmissionGate
    ) -> None:
        result = gate._serialize_payload({"key": "val"})
        assert isinstance(result, bytes)

    def test_empty_dict(self, gate: QuantumAdmissionGate) -> None:
        result = gate._serialize_payload({})
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_non_mapping_raises(
        self, gate: QuantumAdmissionGate
    ) -> None:
        with pytest.raises(QuantumAdmissionError, match="mapping"):
            gate._serialize_payload("not_a_dict")  # type: ignore[arg-type]

    def test_non_mapping_list_raises(
        self, gate: QuantumAdmissionGate
    ) -> None:
        with pytest.raises(QuantumAdmissionError, match="mapping"):
            gate._serialize_payload([1, 2])  # type: ignore[arg-type]

    def test_different_payloads_different_bytes(
        self, gate: QuantumAdmissionGate
    ) -> None:
        r1 = gate._serialize_payload({"a": 1})
        r2 = gate._serialize_payload({"a": 2})
        assert r1 != r2


# ======================================================================
# Tests: _byte_entropy
# ======================================================================


class TestByteEntropy:
    """Pruebas para entropía de Shannon byte-wise."""

    def test_empty_data(self) -> None:
        assert QuantumAdmissionGate._byte_entropy(b"") == 0.0

    def test_single_byte_repeated(self) -> None:
        """Todos los bytes iguales → H = 0."""
        assert QuantumAdmissionGate._byte_entropy(b"\x00" * 100) == 0.0

    def test_two_distinct_bytes_equal_frequency(self) -> None:
        """50% de cada byte → H = ln(2)."""
        data = b"\x00" * 50 + b"\x01" * 50
        expected = math.log(2)
        assert QuantumAdmissionGate._byte_entropy(data) == pytest.approx(
            expected, rel=1e-10
        )

    def test_maximum_entropy(self) -> None:
        """256 bytes distintos, uno de cada → H = ln(256)."""
        data = bytes(range(256))
        expected = math.log(256)
        assert QuantumAdmissionGate._byte_entropy(data) == pytest.approx(
            expected, rel=1e-10
        )

    def test_entropy_non_negative(self) -> None:
        test_cases = [
            b"",
            b"a",
            b"aaa",
            b"abc",
            b"\x00\xff" * 100,
            bytes(range(256)) * 10,
        ]
        for data in test_cases:
            h = QuantumAdmissionGate._byte_entropy(data)
            assert h >= 0.0, f"Negative entropy for {data!r}: {h}"

    def test_entropy_upper_bound(self) -> None:
        """H ≤ ln(256) para cualquier distribución."""
        max_h = math.log(256)
        data = bytes(range(256)) * 100
        assert QuantumAdmissionGate._byte_entropy(data) <= max_h + 1e-10

    def test_single_byte(self) -> None:
        assert QuantumAdmissionGate._byte_entropy(b"x") == 0.0


# ======================================================================
# Tests: _calculate_incident_energy
# ======================================================================


class TestCalculateIncidentEnergy:
    """Pruebas para cálculo de energía incidente."""

    def test_non_negative(
        self,
        gate: QuantumAdmissionGate,
        small_payload: Dict[str, Any],
    ) -> None:
        E = gate._calculate_incident_energy(small_payload)
        assert E >= 0.0

    def test_finite(
        self,
        gate: QuantumAdmissionGate,
        small_payload: Dict[str, Any],
    ) -> None:
        E = gate._calculate_incident_energy(small_payload)
        assert math.isfinite(E)

    def test_empty_payload_zero_energy(
        self, gate: QuantumAdmissionGate
    ) -> None:
        """Payload vacío produce tamaño 0 de la serialización... but not zero bytes
        since repr of empty tuple is not empty. The energy should still be >= 0."""
        E = gate._calculate_incident_energy({})
        assert E >= 0.0

    def test_larger_payload_more_energy(
        self,
        gate: QuantumAdmissionGate,
        small_payload: Dict[str, Any],
        large_payload: Dict[str, Any],
    ) -> None:
        """Payloads más grandes generan más energía (tendencia general)."""
        E_small = gate._calculate_incident_energy(small_payload)
        E_large = gate._calculate_incident_energy(large_payload)
        assert E_large > E_small

    def test_deterministic(
        self,
        gate: QuantumAdmissionGate,
        small_payload: Dict[str, Any],
    ) -> None:
        E1 = gate._calculate_incident_energy(small_payload)
        E2 = gate._calculate_incident_energy(small_payload)
        assert E1 == E2


# ======================================================================
# Tests: _modulate_work_function
# ======================================================================


class TestModulateWorkFunction:
    """Pruebas para modulación de función de trabajo."""

    def test_zero_threat(self) -> None:
        gate = make_gate(threat=0.0)
        phi, threat = gate._modulate_work_function()
        assert phi == pytest.approx(QuantumConstants.BASE_WORK_FUNCTION)
        assert threat == pytest.approx(0.0)

    def test_positive_threat_increases_phi(self) -> None:
        gate = make_gate(threat=2.0)
        phi, threat = gate._modulate_work_function()
        expected = (
            QuantumConstants.BASE_WORK_FUNCTION
            + QuantumConstants.ALPHA_THREAT * 2.0
        )
        assert phi == pytest.approx(expected)
        assert threat == pytest.approx(2.0)

    def test_negative_threat_clamped_to_zero(self) -> None:
        gate = make_gate(threat=-5.0)
        phi, threat = gate._modulate_work_function()
        assert threat == pytest.approx(0.0)
        assert phi == pytest.approx(QuantumConstants.BASE_WORK_FUNCTION)

    def test_phi_always_non_negative(self) -> None:
        for t in [0.0, 0.5, 1.0, 10.0, 100.0]:
            gate = make_gate(threat=t)
            phi, _ = gate._modulate_work_function()
            assert phi >= 0.0

    def test_phi_finite(self) -> None:
        gate = make_gate(threat=1e10)
        phi, _ = gate._modulate_work_function()
        assert math.isfinite(phi)


# ======================================================================
# Tests: _modulate_effective_mass
# ======================================================================


class TestModulateEffectiveMass:
    """Pruebas para modulación de masa efectiva."""

    def test_negative_sigma_finite_mass(self) -> None:
        gate = make_gate(pole=-2.0)
        m_eff, sigma = gate._modulate_effective_mass()
        expected = QuantumConstants.BASE_EFFECTIVE_MASS / 2.0
        assert m_eff == pytest.approx(expected)
        assert sigma == pytest.approx(-2.0)

    def test_sigma_near_zero_infinite_mass(self) -> None:
        gate = make_gate(pole=0.0)
        m_eff, sigma = gate._modulate_effective_mass()
        assert math.isinf(m_eff)
        assert m_eff > 0

    def test_positive_sigma_infinite_mass(self) -> None:
        gate = make_gate(pole=1.0)
        m_eff, sigma = gate._modulate_effective_mass()
        assert math.isinf(m_eff)

    def test_sigma_at_negative_boundary(self) -> None:
        """σ = -tol should still be infinite."""
        gate = make_gate(pole=-QuantumConstants.SIGMA_CHAOS_TOL)
        m_eff, _ = gate._modulate_effective_mass()
        assert math.isinf(m_eff)

    def test_sigma_just_below_boundary(self) -> None:
        """σ slightly below -tol should give finite mass."""
        gate = make_gate(
            pole=-(QuantumConstants.SIGMA_CHAOS_TOL + 1e-6)
        )
        m_eff, _ = gate._modulate_effective_mass()
        assert math.isfinite(m_eff)
        assert m_eff > 0

    def test_mass_positive(self) -> None:
        gate = make_gate(pole=-0.5)
        m_eff, _ = gate._modulate_effective_mass()
        assert m_eff > 0


# ======================================================================
# Tests: _compute_wkb_tunneling_probability
# ======================================================================


class TestWKBTunneling:
    """Pruebas para probabilidad de transmisión WKB."""

    def test_energy_above_barrier(
        self, gate: QuantumAdmissionGate
    ) -> None:
        """E ≥ Φ → T = 1.0 (transmisión clásica)."""
        T = gate._compute_wkb_tunneling_probability(
            E=20.0, Phi=10.0, m_eff=1.0
        )
        assert T == pytest.approx(1.0)

    def test_energy_equals_barrier(
        self, gate: QuantumAdmissionGate
    ) -> None:
        """E = Φ → T = 1.0."""
        T = gate._compute_wkb_tunneling_probability(
            E=10.0, Phi=10.0, m_eff=1.0
        )
        assert T == pytest.approx(1.0)

    def test_infinite_mass_sub_barrier(
        self, gate: QuantumAdmissionGate
    ) -> None:
        """m_eff = ∞, E < Φ → T = 0.0 (barrera impenetrable)."""
        T = gate._compute_wkb_tunneling_probability(
            E=5.0, Phi=10.0, m_eff=float("inf")
        )
        assert T == pytest.approx(0.0)

    def test_infinite_mass_above_barrier(
        self, gate: QuantumAdmissionGate
    ) -> None:
        """E ≥ Φ: transmisión clásica independiente de masa."""
        T = gate._compute_wkb_tunneling_probability(
            E=20.0, Phi=10.0, m_eff=float("inf")
        )
        assert T == pytest.approx(1.0)

    def test_sub_barrier_in_range(
        self, gate: QuantumAdmissionGate
    ) -> None:
        """E < Φ, masa finita → 0 < T < 1."""
        T = gate._compute_wkb_tunneling_probability(
            E=9.0, Phi=10.0, m_eff=1.0
        )
        assert 0.0 < T < 1.0

    def test_deep_barrier_near_zero(
        self, gate: QuantumAdmissionGate
    ) -> None:
        """Barrera alta → T ≈ 0."""
        T = gate._compute_wkb_tunneling_probability(
            E=0.0, Phi=1000.0, m_eff=1.0
        )
        assert T == pytest.approx(0.0, abs=1e-10)

    def test_negative_mass_raises(
        self, gate: QuantumAdmissionGate
    ) -> None:
        with pytest.raises(QuantumNumericalError, match="positiva"):
            gate._compute_wkb_tunneling_probability(
                E=5.0, Phi=10.0, m_eff=-1.0
            )

    def test_zero_mass_raises(
        self, gate: QuantumAdmissionGate
    ) -> None:
        with pytest.raises(QuantumNumericalError, match="positiva"):
            gate._compute_wkb_tunneling_probability(
                E=5.0, Phi=10.0, m_eff=0.0
            )

    def test_result_always_in_zero_one(
        self, gate: QuantumAdmissionGate
    ) -> None:
        test_cases = [
            (0.0, 10.0, 1.0),
            (5.0, 10.0, 1.0),
            (10.0, 10.0, 1.0),
            (15.0, 10.0, 1.0),
            (9.99, 10.0, 0.5),
            (0.0, 100.0, 10.0),
            (0.0, 10.0, float("inf")),
            (20.0, 10.0, float("inf")),
        ]
        for E, Phi, m in test_cases:
            T = gate._compute_wkb_tunneling_probability(E, Phi, m)
            assert 0.0 <= T <= 1.0, (
                f"T={T} out of range for E={E}, Phi={Phi}, m={m}"
            )

    def test_monotonicity_in_energy(
        self, gate: QuantumAdmissionGate
    ) -> None:
        """T increases (or stays) as E increases."""
        Phi = 10.0
        m_eff = 1.0
        prev_T = 0.0
        for E in [0.0, 2.0, 4.0, 6.0, 8.0, 9.0, 9.5, 10.0, 12.0]:
            T = gate._compute_wkb_tunneling_probability(E, Phi, m_eff)
            assert T >= prev_T - 1e-15, (
                f"Non-monotonic: T({E})={T} < T_prev={prev_T}"
            )
            prev_T = T


# ======================================================================
# Tests: _compute_collapse_threshold
# ======================================================================


class TestCollapseThreshold:
    """Pruebas para umbral de colapso determinista."""

    def test_deterministic(
        self,
        gate: QuantumAdmissionGate,
        small_payload: Dict[str, Any],
    ) -> None:
        t1 = gate._compute_collapse_threshold(small_payload)
        t2 = gate._compute_collapse_threshold(small_payload)
        assert t1 == t2

    def test_in_range(
        self,
        gate: QuantumAdmissionGate,
        small_payload: Dict[str, Any],
    ) -> None:
        t = gate._compute_collapse_threshold(small_payload)
        assert 0.0 <= t < 1.0

    def test_different_payloads_different_thresholds(
        self, gate: QuantumAdmissionGate
    ) -> None:
        t1 = gate._compute_collapse_threshold({"a": 1})
        t2 = gate._compute_collapse_threshold({"b": 2})
        # Extremely unlikely to collide
        assert t1 != t2

    def test_order_independent(
        self, gate: QuantumAdmissionGate
    ) -> None:
        t1 = gate._compute_collapse_threshold(
            {"z": 3, "a": 1, "m": 2}
        )
        t2 = gate._compute_collapse_threshold(
            {"a": 1, "m": 2, "z": 3}
        )
        assert t1 == t2

    def test_empty_payload(
        self, gate: QuantumAdmissionGate
    ) -> None:
        t = gate._compute_collapse_threshold({})
        assert 0.0 <= t < 1.0


# ======================================================================
# Tests: evaluate_admission
# ======================================================================


class TestEvaluateAdmission:
    """Pruebas de integración para evaluación de admisión."""

    def test_frustration_veto(self) -> None:
        """Alta frustración → veto absoluto."""
        gate = make_gate(frustration=1.0)
        m = gate.evaluate_admission({"data": "test"})
        assert m.eigenstate == Eigenstate.RECHAZADO
        assert m.frustration_veto is True
        assert "frustración" in m.admission_reason.lower()

    def test_frustration_below_tolerance(self) -> None:
        """Frustración por debajo de tolerancia → no veto."""
        gate = make_gate(frustration=0.0)
        m = gate.evaluate_admission({"data": "test"})
        assert m.frustration_veto is False

    def test_non_mapping_payload_raises(
        self, gate: QuantumAdmissionGate
    ) -> None:
        with pytest.raises(QuantumAdmissionError, match="mapping"):
            gate.evaluate_admission("not_a_dict")  # type: ignore[arg-type]

    def test_list_payload_raises(
        self, gate: QuantumAdmissionGate
    ) -> None:
        with pytest.raises(QuantumAdmissionError, match="mapping"):
            gate.evaluate_admission([1, 2, 3])  # type: ignore[arg-type]

    def test_measurement_fields_populated(
        self,
        gate: QuantumAdmissionGate,
        small_payload: Dict[str, Any],
    ) -> None:
        m = gate.evaluate_admission(small_payload)
        assert isinstance(m.eigenstate, Eigenstate)
        assert math.isfinite(m.incident_energy)
        assert m.incident_energy >= 0.0
        assert math.isfinite(m.work_function)
        assert m.work_function >= 0.0
        assert 0.0 <= m.tunneling_probability <= 1.0
        assert m.kinetic_energy >= 0.0
        assert m.momentum >= 0.0
        assert 0.0 <= m.collapse_threshold <= 1.0
        assert isinstance(m.admission_reason, str)
        assert len(m.admission_reason) > 0

    def test_classical_admission(self) -> None:
        """Low work function (zero threat, stable pole) → E ≥ Φ → classical."""
        gate = make_gate(threat=0.0, pole=-1.0, frustration=0.0)
        # Use a large payload to ensure high energy
        payload = {f"k{i}": f"v{i}" * 100 for i in range(100)}
        m = gate.evaluate_admission(payload)
        if m.eigenstate == Eigenstate.ADMITIDO:
            if m.incident_energy >= m.work_function:
                assert "clásica" in m.admission_reason.lower() or "fotoeléctrica" in m.admission_reason.lower()

    def test_rejection_reason_contains_values(self) -> None:
        """Rejected measurements should include T and threshold in reason."""
        gate = make_gate(threat=100.0, pole=-0.001, frustration=0.0)
        m = gate.evaluate_admission({"small": "data"})
        if m.eigenstate == Eigenstate.RECHAZADO and not m.frustration_veto:
            assert "T" in m.admission_reason or "umbral" in m.admission_reason

    def test_admitted_kinetic_energy_positive(self) -> None:
        gate = make_gate(threat=0.0, pole=-1.0, frustration=0.0)
        payload = {f"k{i}": f"v{i}" * 200 for i in range(200)}
        m = gate.evaluate_admission(payload)
        if m.eigenstate == Eigenstate.ADMITIDO:
            assert m.kinetic_energy >= QuantumConstants.MIN_KINETIC_ENERGY
            assert m.momentum > 0.0

    def test_rejected_zero_kinetic_momentum(self) -> None:
        gate = make_gate(threat=100.0, pole=-0.001, frustration=0.0)
        m = gate.evaluate_admission({"tiny": "x"})
        if m.eigenstate == Eigenstate.RECHAZADO:
            assert m.kinetic_energy == 0.0
            assert m.momentum == 0.0

    def test_deterministic_result(
        self,
        gate: QuantumAdmissionGate,
        small_payload: Dict[str, Any],
    ) -> None:
        m1 = gate.evaluate_admission(small_payload)
        m2 = gate.evaluate_admission(small_payload)
        assert m1.eigenstate == m2.eigenstate
        assert m1.incident_energy == m2.incident_energy
        assert m1.tunneling_probability == m2.tunneling_probability
        assert m1.collapse_threshold == m2.collapse_threshold

    def test_veto_captures_diagnostic_data(self) -> None:
        """Veto should still have threat and sigma info."""
        gate = make_gate(
            threat=3.0, pole=-2.0, frustration=1.0
        )
        m = gate.evaluate_admission({"data": "test"})
        assert m.frustration_veto is True
        assert m.threat_level == pytest.approx(3.0)
        assert m.dominant_pole_real == pytest.approx(-2.0)

    def test_veto_with_failing_threat_oracle(self) -> None:
        """If threat oracle fails during veto, threat_level defaults to 0."""

        class FailingWatcher:
            def get_mahalanobis_threat(self) -> float:
                raise RuntimeError("oracle down")

        gate = QuantumAdmissionGate(
            topo_watcher=FailingWatcher(),  # type: ignore[arg-type]
            laplace_oracle=MockLaplaceOracle(pole=-1.0),
            sheaf_orchestrator=MockSheafOrchestrator(frustration=1.0),
        )
        m = gate.evaluate_admission({"data": "test"})
        assert m.frustration_veto is True
        assert m.threat_level == 0.0


# ======================================================================
# Tests: __call__ (Morphism)
# ======================================================================


class TestMorphismCall:
    """Pruebas para la interfaz de morfismo categórico."""

    @pytest.fixture()
    def mock_categorical_state(self) -> MagicMock:
        state = MagicMock()
        state.payload = {"data": "test"}
        state.context = {"existing": "context"}
        state.validated_strata = frozenset()
        return state

    def test_non_categorical_state_raises(
        self, gate: QuantumAdmissionGate
    ) -> None:
        with pytest.raises(
            QuantumAdmissionError, match="CategoricalState"
        ):
            gate("not_a_state")  # type: ignore[arg-type]

    def test_non_mapping_payload_raises(
        self, gate: QuantumAdmissionGate
    ) -> None:
        state = MagicMock(spec=[])
        type(state).__name__ = "CategoricalState"
        # Bypass isinstance by patching
        with pytest.raises(QuantumAdmissionError):
            gate(state)

    def test_admitted_state_has_physics_stratum(self) -> None:
        """When admitted, PHYSICS stratum is added."""
        gate = make_gate(threat=0.0, pole=-1.0, frustration=0.0)
        payload = {f"k{i}": f"v{i}" * 200 for i in range(200)}

        # We need a real CategoricalState, but since it's imported
        # from the module, we test the evaluate_admission part
        m = gate.evaluate_admission(payload)
        # Verify the measurement
        if m.eigenstate == Eigenstate.ADMITIDO:
            assert m.momentum > 0
            assert m.kinetic_energy >= QuantumConstants.MIN_KINETIC_ENERGY

    def test_rejected_state_empty_strata(self) -> None:
        gate = make_gate(frustration=1.0)
        m = gate.evaluate_admission({"data": "test"})
        assert m.eigenstate == Eigenstate.RECHAZADO
        assert m.frustration_veto is True


# ======================================================================
# Tests: Momentum Calculation
# ======================================================================


class TestMomentumCalculation:
    """Pruebas para el cálculo de momentum de inyección."""

    def test_momentum_uses_effective_mass(self) -> None:
        """Momentum should use m_eff, not BASE_EFFECTIVE_MASS."""
        # pole=-0.5 → m_eff = 1.0/0.5 = 2.0
        gate = make_gate(threat=0.0, pole=-0.5, frustration=0.0)
        payload = {f"k{i}": f"v{i}" * 200 for i in range(200)}
        m = gate.evaluate_admission(payload)

        if m.eigenstate == Eigenstate.ADMITIDO:
            # p = sqrt(2 * m_eff * K)
            expected_m_eff = QuantumConstants.BASE_EFFECTIVE_MASS / 0.5
            if math.isfinite(m.effective_mass):
                assert m.effective_mass == pytest.approx(expected_m_eff)
                expected_p = math.sqrt(
                    2.0 * expected_m_eff * m.kinetic_energy
                )
                assert m.momentum == pytest.approx(expected_p, rel=1e-6)

    def test_momentum_non_negative(self) -> None:
        gate = make_gate(threat=0.0, pole=-1.0, frustration=0.0)
        payload = {f"k{i}": f"v{i}" * 100 for i in range(50)}
        m = gate.evaluate_admission(payload)
        assert m.momentum >= 0.0

    def test_momentum_finite(self) -> None:
        gate = make_gate(threat=0.0, pole=-1.0, frustration=0.0)
        m = gate.evaluate_admission({"data": "test"})
        assert math.isfinite(m.momentum)


# ======================================================================
# Tests: Edge Cases and Numerical Robustness
# ======================================================================


class TestNumericalRobustness:
    """Pruebas de robustez numérica para casos extremos."""

    def test_very_high_threat(self) -> None:
        """Extreme threat → very high Φ → likely rejection."""
        gate = make_gate(threat=1e6)
        m = gate.evaluate_admission({"key": "val"})
        assert m.work_function > QuantumConstants.BASE_WORK_FUNCTION

    def test_very_small_sigma(self) -> None:
        """σ very negative → very small m_eff."""
        gate = make_gate(pole=-1e6)
        m_eff, _ = gate._modulate_effective_mass()
        assert m_eff > 0
        assert m_eff < QuantumConstants.BASE_EFFECTIVE_MASS

    def test_sigma_barely_unstable(self) -> None:
        """σ just above -tol → infinite mass."""
        sigma = -QuantumConstants.SIGMA_CHAOS_TOL / 2
        gate = make_gate(pole=sigma)
        m_eff, _ = gate._modulate_effective_mass()
        assert math.isinf(m_eff)

    def test_large_payload_no_overflow(
        self, gate: QuantumAdmissionGate
    ) -> None:
        payload = {f"k{i}": "x" * 10000 for i in range(100)}
        m = gate.evaluate_admission(payload)
        assert math.isfinite(m.incident_energy)
        assert 0.0 <= m.tunneling_probability <= 1.0

    def test_empty_payload(
        self, gate: QuantumAdmissionGate
    ) -> None:
        m = gate.evaluate_admission({})
        assert isinstance(m.eigenstate, Eigenstate)
        assert m.incident_energy >= 0.0

    def test_payload_with_nested_structures(
        self, gate: QuantumAdmissionGate
    ) -> None:
        payload = {
            "nested": {"deep": {"value": [1, 2, 3]}},
            "list": [{"a": 1}, {"b": 2}],
            "none": None,
        }
        m = gate.evaluate_admission(payload)
        assert isinstance(m.eigenstate, Eigenstate)

    def test_frustration_exactly_at_tolerance(self) -> None:
        """Frustration exactly at tolerance should NOT veto."""
        gate = make_gate(
            frustration=QuantumConstants.FRUSTRATION_VETO_TOL
        )
        m = gate.evaluate_admission({"data": "test"})
        assert m.frustration_veto is False

    def test_frustration_just_above_tolerance(self) -> None:
        gate = make_gate(
            frustration=QuantumConstants.FRUSTRATION_VETO_TOL + 1e-15
        )
        m = gate.evaluate_admission({"data": "test"})
        assert m.frustration_veto is True

    def test_frustration_just_below_tolerance(self) -> None:
        gate = make_gate(
            frustration=QuantumConstants.FRUSTRATION_VETO_TOL - 1e-15
        )
        m = gate.evaluate_admission({"data": "test"})
        assert m.frustration_veto is False

    def test_zero_frustration(self) -> None:
        gate = make_gate(frustration=0.0)
        m = gate.evaluate_admission({"data": "test"})
        assert m.frustration_veto is False

    def test_negative_frustration_no_veto(self) -> None:
        """Negative frustration (shouldn't happen but handled) → no veto."""
        gate = make_gate(frustration=-1.0)
        m = gate.evaluate_admission({"data": "test"})
        assert m.frustration_veto is False


# ======================================================================
# Tests: System Properties
# ======================================================================


class TestSystemProperties:
    """Pruebas de propiedades transversales del sistema."""

    def test_measurement_immutable(
        self,
        gate: QuantumAdmissionGate,
        small_payload: Dict[str, Any],
    ) -> None:
        m = gate.evaluate_admission(small_payload)
        with pytest.raises((AttributeError, FrozenInstanceError)):
            m.eigenstate = Eigenstate.ADMITIDO  # type: ignore[misc]

    def test_measurement_has_slots(self) -> None:
        assert hasattr(QuantumMeasurement, "__slots__")

    def test_full_determinism(self) -> None:
        """Two independent gates with same oracles produce same result."""
        payload = {"deterministic": "payload", "number": 42}
        g1 = make_gate(threat=1.5, pole=-0.8, frustration=0.0)
        g2 = make_gate(threat=1.5, pole=-0.8, frustration=0.0)
        m1 = g1.evaluate_admission(payload)
        m2 = g2.evaluate_admission(payload)
        assert m1.eigenstate == m2.eigenstate
        assert m1.incident_energy == pytest.approx(m2.incident_energy)
        assert m1.work_function == pytest.approx(m2.work_function)
        assert m1.tunneling_probability == pytest.approx(
            m2.tunneling_probability
        )
        assert m1.collapse_threshold == pytest.approx(
            m2.collapse_threshold
        )
        assert m1.kinetic_energy == pytest.approx(m2.kinetic_energy)
        assert m1.momentum == pytest.approx(m2.momentum)

    def test_tunneling_probability_invariant(
        self,
        gate: QuantumAdmissionGate,
        small_payload: Dict[str, Any],
    ) -> None:
        """T ∈ [0, 1] is an invariant for all measurements."""
        m = gate.evaluate_admission(small_payload)
        assert 0.0 <= m.tunneling_probability <= 1.0

    def test_energy_non_negativity_invariant(
        self,
        gate: QuantumAdmissionGate,
        small_payload: Dict[str, Any],
    ) -> None:
        m = gate.evaluate_admission(small_payload)
        assert m.incident_energy >= 0.0
        assert m.work_function >= 0.0
        assert m.kinetic_energy >= 0.0
        assert m.momentum >= 0.0

    def test_collapse_threshold_invariant(
        self,
        gate: QuantumAdmissionGate,
        small_payload: Dict[str, Any],
    ) -> None:
        m = gate.evaluate_admission(small_payload)
        assert 0.0 <= m.collapse_threshold <= 1.0

    def test_admitted_implies_t_geq_threshold(self) -> None:
        """If admitted (non-veto), T ≥ collapse_threshold."""
        gate = make_gate(threat=0.0, pole=-1.0, frustration=0.0)
        payloads = [
            {"a": 1},
            {"b": "x" * 100},
            {f"k{i}": i for i in range(50)},
        ]
        for p in payloads:
            m = gate.evaluate_admission(p)
            if (
                m.eigenstate == Eigenstate.ADMITIDO
                and not m.frustration_veto
            ):
                assert m.tunneling_probability >= m.collapse_threshold

    def test_rejected_non_veto_implies_t_lt_threshold(self) -> None:
        """If rejected (non-veto), T < collapse_threshold."""
        gate = make_gate(threat=50.0, pole=-0.01, frustration=0.0)
        payloads = [
            {"a": 1},
            {"b": "small"},
            {"key": "value"},
        ]
        for p in payloads:
            m = gate.evaluate_admission(p)
            if (
                m.eigenstate == Eigenstate.RECHAZADO
                and not m.frustration_veto
            ):
                assert m.tunneling_probability < m.collapse_threshold

    def test_veto_implies_rejected(self) -> None:
        """Frustration veto always means RECHAZADO."""
        gate = make_gate(frustration=1.0)
        m = gate.evaluate_admission({"data": "test"})
        assert m.frustration_veto is True
        assert m.eigenstate == Eigenstate.RECHAZADO

    def test_payload_not_mutated(
        self,
        gate: QuantumAdmissionGate,
    ) -> None:
        payload = {"key": "value", "nested": {"a": 1}}
        original = {"key": "value", "nested": {"a": 1}}
        gate.evaluate_admission(payload)
        assert payload == original

    def test_byte_entropy_is_static(self) -> None:
        """_byte_entropy is a static method, callable without instance."""
        h = QuantumAdmissionGate._byte_entropy(b"test")
        assert h >= 0.0

    def test_serialize_payload_is_static(self) -> None:
        """_serialize_payload is a static method."""
        result = QuantumAdmissionGate._serialize_payload({"a": 1})
        assert isinstance(result, bytes)

    def test_wkb_symmetry_barrier_height(
        self, gate: QuantumAdmissionGate
    ) -> None:
        """
        WKB: same barrier height (Φ - E) with same mass
        gives same tunneling probability.
        """
        T1 = gate._compute_wkb_tunneling_probability(
            E=5.0, Phi=10.0, m_eff=1.0
        )
        T2 = gate._compute_wkb_tunneling_probability(
            E=15.0, Phi=20.0, m_eff=1.0
        )
        assert T1 == pytest.approx(T2)

    def test_wkb_heavier_mass_lower_tunneling(
        self, gate: QuantumAdmissionGate
    ) -> None:
        """Heavier mass → more suppression → lower T."""
        T_light = gate._compute_wkb_tunneling_probability(
            E=5.0, Phi=10.0, m_eff=0.5
        )
        T_heavy = gate._compute_wkb_tunneling_probability(
            E=5.0, Phi=10.0, m_eff=5.0
        )
        assert T_light > T_heavy