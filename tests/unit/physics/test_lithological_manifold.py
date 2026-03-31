"""
Suite de pruebas: test_lithological_manifold.py
Ubicación: tests/physics/test_lithological_manifold.py

Descripción
-----------
Batería exhaustiva de pruebas para el módulo LithologicalManifold.
Organizada en clases temáticas que cubren:

    §1  Funciones auxiliares puras.
    §2  Validación del SoilTensor (invariantes de Casagrande, rangos, finitud).
    §3  Magnitudes derivadas (propiedades matemáticas, dimensional, monotonía).
    §4  Reglas diagnósticas y emisión de cartuchos.
    §5  Singularidades litológicas (turba).
    §6  Morfismo completo (__call__) — integración.
    §7  Propiedades de normalización y acotamiento [0, 1].
    §8  Robustez numérica y casos extremos.
    §9  Serialización y trazabilidad.

Principios
----------
- Cada test verifica exactamente una propiedad o invariante.
- Los nombres siguen la convención: test_<qué>_<condición>_<resultado>.
- Se evita dependencia entre tests (cada uno construye su estado).
- Se parametrizan los casos donde la variación es dimensional.
- Se documentan las propiedades matemáticas verificadas.

Dependencias de mock
--------------------
Se mockean las importaciones externas (Morphism, CategoricalState, Stratum,
MetricTensorFactory) para aislar la lógica bajo prueba del framework MIC.
"""

from __future__ import annotations

import math
import sys
import types
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Tuple
from unittest.mock import MagicMock

import pytest


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              BOOTSTRAP: MOCKS DE DEPENDENCIAS EXTERNAS                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _bootstrap_external_mocks() -> None:
    """
    Inyecta módulos mock en sys.modules para que el import del módulo
    bajo prueba no falle por dependencias externas al framework MIC.

    Esto permite ejecutar los tests en aislamiento sin necesidad de
    instalar todo el framework.
    """
    mock_schemas = MagicMock()
    # Simulación del retículo de estratos
    mock_schemas.Stratum.PHYSICS = 4
    sys.modules["app.core.schemas"] = mock_schemas

    mock_mic_algebra = MagicMock()
    mock_mic_algebra.CategoricalState = MagicMock
    mock_mic_algebra.Morphism = MagicMock
    sys.modules["app.core.mic_algebra"] = mock_mic_algebra

    mock_metric_tensors = MagicMock()
    sys.modules["app.core.immune_system.metric_tensors"] = mock_metric_tensors


# Ejecutar bootstrap ANTES de importar el módulo bajo prueba
_bootstrap_external_mocks()

# --- Importaciones del módulo bajo prueba ---
from app.physics.lithological_manifold import (
    GeomechanicalConstants,
    DiagnosticRule,
    LithologicalManifold,
    LithologicalInputError,
    LithologicalManifoldError,
    LithologicalNumericalError,
    LithologicalSingularityError,
    LithologyDiagnosticReport,
    LiquefactionSolitonCartridge,
    SoilTensor,
    SwellingPlasmonCartridge,
    YieldingPhononCartridge,
    _clamp,
    _is_fine_grained,
    _is_peat,
    _is_sand_like,
    _normalize_uscs,
    _parse_bool,
    _safe_divide,
    _safe_upper_strip,
    _sigmoid,
    _assert_finite,
    _assert_unit_interval,
    USCS_ALL_VALID_GROUPS,
    USCS_COARSE_GROUPS,
    USCS_FINE_GROUPS,
    USCS_ORGANIC_EXTREME_GROUPS,
)

# Alias para brevedad
C = GeomechanicalConstants


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    FIXTURES REUTILIZABLES                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@pytest.fixture
def manifold() -> LithologicalManifold:
    """Instancia fresca del operador litológico."""
    return LithologicalManifold()


@pytest.fixture
def nominal_sand_tensor() -> SoilTensor:
    """Arena limpia, no saturada, propiedades centrales."""
    return SoilTensor(
        uscs_classification="SW",
        liquid_limit=0.0,
        plasticity_index=0.0,
        shear_wave_velocity=300.0,
        void_ratio=0.5,
        is_saturated=False,
        bulk_density_kg_m3=1800.0,
    )


@pytest.fixture
def expansive_clay_tensor() -> SoilTensor:
    """Arcilla expansiva de alta plasticidad."""
    return SoilTensor(
        uscs_classification="CH",
        liquid_limit=75.0,
        plasticity_index=40.0,
        shear_wave_velocity=180.0,
        void_ratio=0.9,
        is_saturated=True,
        bulk_density_kg_m3=1650.0,
    )


@pytest.fixture
def soft_clay_tensor() -> SoilTensor:
    """Arcilla blanda con alto void ratio y baja Vs."""
    return SoilTensor(
        uscs_classification="CL",
        liquid_limit=45.0,
        plasticity_index=20.0,
        shear_wave_velocity=120.0,
        void_ratio=1.2,
        is_saturated=True,
        bulk_density_kg_m3=1550.0,
    )


@pytest.fixture
def liquefiable_sand_tensor() -> SoilTensor:
    """Arena saturada con Vs por debajo del umbral de licuación."""
    return SoilTensor(
        uscs_classification="SP",
        liquid_limit=0.0,
        plasticity_index=0.0,
        shear_wave_velocity=100.0,
        void_ratio=0.7,
        is_saturated=True,
        bulk_density_kg_m3=1900.0,
    )


@pytest.fixture
def peat_tensor() -> SoilTensor:
    """Turba — singularidad litológica."""
    return SoilTensor(
        uscs_classification="PT",
        liquid_limit=200.0,
        plasticity_index=100.0,
        shear_wave_velocity=60.0,
        void_ratio=8.0,
        is_saturated=True,
        bulk_density_kg_m3=1050.0,
    )


@pytest.fixture
def nominal_payload() -> Dict[str, Any]:
    """Payload externo nominal para arena."""
    return {
        "uscs": "SW",
        "liquid_limit": 0.0,
        "plasticity_index": 0.0,
        "vs": 300.0,
        "void_ratio": 0.5,
        "is_saturated": False,
        "bulk_density_kg_m3": 1800.0,
    }


@pytest.fixture
def empty_context() -> Dict[str, Any]:
    """Contexto vacío."""
    return {}


def _make_payload(**overrides: Any) -> Dict[str, Any]:
    """Fábrica de payloads con sobrecargas puntuales."""
    base = {
        "uscs": "SW",
        "liquid_limit": 25.0,
        "plasticity_index": 5.0,
        "vs": 300.0,
        "void_ratio": 0.5,
        "is_saturated": False,
        "bulk_density_kg_m3": 1800.0,
    }
    base.update(overrides)
    return base


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              §1  FUNCIONES AUXILIARES PURAS                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class TestSafeUpperStrip:
    """Pruebas para _safe_upper_strip."""

    def test_basic_normalization(self) -> None:
        assert _safe_upper_strip("  sw  ") == "SW"

    def test_already_upper(self) -> None:
        assert _safe_upper_strip("CH") == "CH"

    def test_mixed_case_with_hyphen(self) -> None:
        assert _safe_upper_strip(" sw-sm ") == "SW-SM"

    def test_empty_string(self) -> None:
        assert _safe_upper_strip("") == ""

    def test_whitespace_only(self) -> None:
        assert _safe_upper_strip("   ") == ""


class TestClamp:
    """Pruebas para _clamp."""

    def test_within_range(self) -> None:
        assert _clamp(0.5, 0.0, 1.0) == 0.5

    def test_at_lower_bound(self) -> None:
        assert _clamp(0.0, 0.0, 1.0) == 0.0

    def test_at_upper_bound(self) -> None:
        assert _clamp(1.0, 0.0, 1.0) == 1.0

    def test_below_lower_bound(self) -> None:
        assert _clamp(-0.5, 0.0, 1.0) == 0.0

    def test_above_upper_bound(self) -> None:
        assert _clamp(1.5, 0.0, 1.0) == 1.0

    def test_degenerate_interval(self) -> None:
        """Cuando lo == hi, retorna ese valor."""
        assert _clamp(5.0, 3.0, 3.0) == 3.0

    def test_clamp_invariants(self) -> None:
        """Prueba casos límites (Inf, -Inf, NaN) sobre el operador de saturación."""
        assert math.isclose(_clamp(float('inf'), 0.0, 1.0), 1.0, abs_tol=1e-12)
        assert math.isclose(_clamp(float('-inf'), 0.0, 1.0), 0.0, abs_tol=1e-12)
        assert math.isnan(_clamp(float('nan'), 0.0, 1.0))


class TestAssertFinite:
    """Pruebas para _assert_finite."""

    def test_finite_value_passes(self) -> None:
        assert _assert_finite(42.0, "test") == 42.0

    def test_zero_passes(self) -> None:
        assert _assert_finite(0.0, "test") == 0.0

    def test_negative_passes(self) -> None:
        assert _assert_finite(-1.0, "test") == -1.0

    def test_nan_raises(self) -> None:
        with pytest.raises(LithologicalNumericalError, match="no finito"):
            _assert_finite(float("nan"), "campo_x")

    def test_positive_inf_raises(self) -> None:
        with pytest.raises(LithologicalNumericalError):
            _assert_finite(float("inf"), "campo_y")

    def test_negative_inf_raises(self) -> None:
        with pytest.raises(LithologicalNumericalError):
            _assert_finite(float("-inf"), "campo_z")


class TestAssertUnitInterval:
    """Pruebas para _assert_unit_interval."""

    def test_zero_passes(self) -> None:
        _assert_unit_interval(0.0, "idx")

    def test_one_passes(self) -> None:
        _assert_unit_interval(1.0, "idx")

    def test_midpoint_passes(self) -> None:
        _assert_unit_interval(0.5, "idx")

    def test_slightly_above_one_within_tolerance(self) -> None:
        """Valores dentro de tolerancia de máquina deben pasar."""
        _assert_unit_interval(1.0 + 1e-15, "idx")

    def test_clearly_above_one_raises(self) -> None:
        with pytest.raises(LithologicalNumericalError, match="fuera de"):
            _assert_unit_interval(1.5, "idx")

    def test_clearly_below_zero_raises(self) -> None:
        with pytest.raises(LithologicalNumericalError, match="fuera de"):
            _assert_unit_interval(-0.5, "idx")


class TestSigmoid:
    """
    Pruebas para _sigmoid.

    Propiedades matemáticas verificadas:
        P1: σ(x₀) = 0.5   (punto medio)
        P2: σ ∈ (0, 1)     (imagen acotada)
        P3: monótona creciente para k > 0
        P4: estabilidad numérica para |z| >> 1
    """

    def test_midpoint_equals_half(self) -> None:
        """P1: σ(x₀; x₀, k) = 0.5 para todo k > 0."""
        assert _sigmoid(10.0, 10.0, 0.5) == pytest.approx(0.5, abs=1e-10)

    def test_large_positive_saturates_to_one(self) -> None:
        """P2/P4: σ(x >> x₀) → 1."""
        result = _sigmoid(1000.0, 10.0, 0.5)
        assert result == pytest.approx(1.0, abs=1e-10)

    def test_large_negative_saturates_to_zero(self) -> None:
        """P2/P4: σ(x << x₀) → 0."""
        result = _sigmoid(-1000.0, 10.0, 0.5)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_monotonically_increasing(self) -> None:
        """P3: x₁ < x₂ ⟹ σ(x₁) < σ(x₂) para k > 0."""
        x_values = [-10.0, 0.0, 5.0, 10.0, 15.0, 20.0, 50.0]
        results = [_sigmoid(x, 10.0, 0.2) for x in x_values]
        for i in range(len(results) - 1):
            assert results[i] < results[i + 1], (
                f"Monotonía violada: σ({x_values[i]}) = {results[i]} "
                f"≥ σ({x_values[i+1]}) = {results[i+1]}"
            )

    def test_output_strictly_in_unit_interval(self) -> None:
        """P2: σ(x) ∈ (0, 1) para todo x finito."""
        for x in [-500.0, -1.0, 0.0, 1.0, 500.0]:
            result = _sigmoid(x, 0.0, 1.0)
            assert 0.0 <= result <= 1.0

    def test_overflow_protection_extreme_positive(self) -> None:
        """P4: No lanza excepción para z extremadamente grande."""
        result = _sigmoid(1e6, 0.0, 1.0)
        assert result == 1.0

    def test_overflow_protection_extreme_negative(self) -> None:
        """P4: No lanza excepción para z extremadamente negativo."""
        result = _sigmoid(-1e6, 0.0, 1.0)
        assert result == 0.0

    def test_symmetry_around_midpoint(self) -> None:
        """σ(x₀ + δ) + σ(x₀ - δ) = 1 (propiedad de simetría logística)."""
        x0, k, delta = 15.0, 0.3, 7.0
        s_plus = _sigmoid(x0 + delta, x0, k)
        s_minus = _sigmoid(x0 - delta, x0, k)
        assert s_plus + s_minus == pytest.approx(1.0, abs=1e-10)


class TestSafeDivide:
    """Pruebas para _safe_divide."""

    def test_normal_division(self) -> None:
        assert _safe_divide(10.0, 2.0) == pytest.approx(5.0)

    def test_zero_denominator_returns_fallback(self) -> None:
        assert _safe_divide(10.0, 0.0, fallback=0.0) == 0.0

    def test_tiny_denominator_returns_fallback(self) -> None:
        assert _safe_divide(10.0, 1e-15, fallback=-1.0) == -1.0

    def test_negative_denominator_near_zero(self) -> None:
        assert _safe_divide(10.0, -1e-15, fallback=42.0) == 42.0

    def test_zero_numerator(self) -> None:
        assert _safe_divide(0.0, 5.0) == 0.0

    def test_custom_fallback(self) -> None:
        assert _safe_divide(1.0, 0.0, fallback=999.0) == 999.0


class TestParseBool:
    """Pruebas para _parse_bool."""

    @pytest.mark.parametrize("value,expected", [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        (1.0, True),
        (0.0, False),
        ("true", True),
        ("TRUE", True),
        ("True", True),
        ("false", False),
        ("FALSE", False),
        ("1", True),
        ("0", False),
        ("yes", True),
        ("no", False),
        ("si", True),
        ("sí", True),
        ("y", True),
        ("n", False),
    ])
    def test_valid_inputs(self, value: Any, expected: bool) -> None:
        assert _parse_bool(value) is expected

    @pytest.mark.parametrize("value", [
        "maybe", "perhaps", "2", "nil", None, [], {},
    ])
    def test_invalid_inputs_raise(self, value: Any) -> None:
        with pytest.raises(LithologicalInputError):
            _parse_bool(value)

    def test_nan_float_raises(self) -> None:
        with pytest.raises(LithologicalInputError):
            _parse_bool(float("nan"))

    def test_inf_float_raises(self) -> None:
        with pytest.raises(LithologicalInputError):
            _parse_bool(float("inf"))


class TestNormalizeUscs:
    """Pruebas para _normalize_uscs."""

    def test_basic(self) -> None:
        assert _normalize_uscs("sw") == "SW"

    def test_dual_group(self) -> None:
        assert _normalize_uscs(" sw-sm ") == "SW-SM"

    def test_preserves_hyphen(self) -> None:
        assert "-" in _normalize_uscs("cl-ml")


class TestUSCSClassifiers:
    """Pruebas para _is_peat, _is_sand_like, _is_fine_grained."""

    @pytest.mark.parametrize("uscs", ["PT", "pt", " Pt "])
    def test_is_peat_positive(self, uscs: str) -> None:
        assert _is_peat(uscs) is True

    @pytest.mark.parametrize("uscs", ["SW", "CH", "ML", "GW"])
    def test_is_peat_negative(self, uscs: str) -> None:
        assert _is_peat(uscs) is False

    @pytest.mark.parametrize("uscs", ["SW", "SP", "SM", "SC"])
    def test_is_sand_like_primary(self, uscs: str) -> None:
        assert _is_sand_like(uscs) is True

    @pytest.mark.parametrize("uscs", ["SW-SM", "SP-SC", "SW-SC"])
    def test_is_sand_like_dual(self, uscs: str) -> None:
        assert _is_sand_like(uscs) is True

    @pytest.mark.parametrize("uscs", ["GW", "CH", "ML", "PT"])
    def test_is_sand_like_negative(self, uscs: str) -> None:
        assert _is_sand_like(uscs) is False

    @pytest.mark.parametrize("uscs", ["ML", "CL", "OL", "MH", "CH", "OH"])
    def test_is_fine_grained_positive(self, uscs: str) -> None:
        assert _is_fine_grained(uscs) is True

    def test_is_fine_grained_dual(self) -> None:
        assert _is_fine_grained("CL-ML") is True

    @pytest.mark.parametrize("uscs", ["SW", "GP", "PT"])
    def test_is_fine_grained_negative(self, uscs: str) -> None:
        assert _is_fine_grained(uscs) is False


class TestUSCSGroupCompleteness:
    """Verificación de completitud de los conjuntos USCS."""

    def test_no_overlap_coarse_fine(self) -> None:
        """Los grupos gruesos y finos deben ser disjuntos."""
        assert USCS_COARSE_GROUPS & USCS_FINE_GROUPS == set()

    def test_no_overlap_coarse_organic(self) -> None:
        assert USCS_COARSE_GROUPS & USCS_ORGANIC_EXTREME_GROUPS == set()

    def test_no_overlap_fine_organic(self) -> None:
        assert USCS_FINE_GROUPS & USCS_ORGANIC_EXTREME_GROUPS == set()

    def test_all_valid_is_union(self) -> None:
        """USCS_ALL_VALID_GROUPS = coarse ∪ fine ∪ organic."""
        expected = (
            USCS_COARSE_GROUPS | USCS_FINE_GROUPS
            | USCS_ORGANIC_EXTREME_GROUPS
        )
        assert USCS_ALL_VALID_GROUPS == expected

    def test_pt_in_organic(self) -> None:
        assert "PT" in USCS_ORGANIC_EXTREME_GROUPS


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              §2  VALIDACIÓN DEL SOIL TENSOR                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class TestSoilTensorValidation:
    """
    Pruebas de SoilTensor.validate().

    Verifica todos los invariantes documentados:
        I1: LL ≥ 0
        I2: PI ≥ 0
        I3: PI ≤ LL  (Casagrande)
        I4: Vs > 0
        I5: e₀ ≥ 0
        I6: ρ > 0
        I7: USCS reconocido
        I8: Todos finitos
        I9: Rangos admisibles
    """

    def test_nominal_sand_passes(self, nominal_sand_tensor: SoilTensor) -> None:
        """Un tensor nominal no debe lanzar excepciones."""
        nominal_sand_tensor.validate()  # No exception

    def test_all_valid_uscs_pass(self) -> None:
        """Todo grupo USCS reconocido debe pasar validación."""
        for uscs in USCS_ALL_VALID_GROUPS:
            tensor = SoilTensor(
                uscs_classification=uscs,
                liquid_limit=30.0,
                plasticity_index=10.0,
                shear_wave_velocity=250.0,
                void_ratio=0.6,
                is_saturated=False,
            )
            tensor.validate()

    # --- I7: Clasificación USCS ---

    def test_empty_uscs_raises(self) -> None:
        tensor = SoilTensor(
            uscs_classification="",
            liquid_limit=30.0,
            plasticity_index=10.0,
            shear_wave_velocity=250.0,
            void_ratio=0.6,
            is_saturated=False,
        )
        with pytest.raises(LithologicalInputError, match="vacía"):
            tensor.validate()

    def test_unknown_uscs_raises(self) -> None:
        tensor = SoilTensor(
            uscs_classification="ZZ",
            liquid_limit=30.0,
            plasticity_index=10.0,
            shear_wave_velocity=250.0,
            void_ratio=0.6,
            is_saturated=False,
        )
        with pytest.raises(LithologicalInputError, match="no reconocida"):
            tensor.validate()

    # --- I1: LL ≥ 0 ---

    def test_negative_liquid_limit_raises(self) -> None:
        tensor = SoilTensor(
            uscs_classification="CL",
            liquid_limit=-5.0,
            plasticity_index=0.0,
            shear_wave_velocity=250.0,
            void_ratio=0.6,
            is_saturated=False,
        )
        with pytest.raises(LithologicalInputError, match="negativo"):
            tensor.validate()

    # --- I2: PI ≥ 0 ---

    def test_negative_plasticity_index_raises(self) -> None:
        tensor = SoilTensor(
            uscs_classification="CL",
            liquid_limit=30.0,
            plasticity_index=-1.0,
            shear_wave_velocity=250.0,
            void_ratio=0.6,
            is_saturated=False,
        )
        with pytest.raises(LithologicalInputError, match="negativo"):
            tensor.validate()

    # --- I3: PI ≤ LL (Casagrande) ---

    def test_pi_exceeds_ll_raises(self) -> None:
        """Restricción de Casagrande: PI > LL es físicamente imposible."""
        tensor = SoilTensor(
            uscs_classification="CH",
            liquid_limit=40.0,
            plasticity_index=50.0,
            shear_wave_velocity=250.0,
            void_ratio=0.6,
            is_saturated=False,
        )
        with pytest.raises(LithologicalInputError, match="Casagrande"):
            tensor.validate()

    def test_pi_equals_ll_passes(self) -> None:
        """PI = LL es teóricamente posible (caso límite)."""
        tensor = SoilTensor(
            uscs_classification="CH",
            liquid_limit=60.0,
            plasticity_index=60.0,
            shear_wave_velocity=250.0,
            void_ratio=0.6,
            is_saturated=False,
        )
        tensor.validate()

    def test_pi_zero_ll_zero_passes(self) -> None:
        """Suelo no plástico: PI = LL = 0."""
        tensor = SoilTensor(
            uscs_classification="SW",
            liquid_limit=0.0,
            plasticity_index=0.0,
            shear_wave_velocity=300.0,
            void_ratio=0.5,
            is_saturated=False,
        )
        tensor.validate()

    # --- I4: Vs > 0 ---

    def test_zero_vs_raises(self) -> None:
        tensor = SoilTensor(
            uscs_classification="SW",
            liquid_limit=0.0,
            plasticity_index=0.0,
            shear_wave_velocity=0.0,
            void_ratio=0.5,
            is_saturated=False,
        )
        with pytest.raises(LithologicalInputError, match="positiva"):
            tensor.validate()

    def test_negative_vs_raises(self) -> None:
        tensor = SoilTensor(
            uscs_classification="SW",
            liquid_limit=0.0,
            plasticity_index=0.0,
            shear_wave_velocity=-100.0,
            void_ratio=0.5,
            is_saturated=False,
        )
        with pytest.raises(LithologicalInputError, match="positiva"):
            tensor.validate()

    # --- I5: e₀ ≥ 0 ---

    def test_negative_void_ratio_raises(self) -> None:
        tensor = SoilTensor(
            uscs_classification="SW",
            liquid_limit=0.0,
            plasticity_index=0.0,
            shear_wave_velocity=300.0,
            void_ratio=-0.1,
            is_saturated=False,
        )
        with pytest.raises(LithologicalInputError, match="negativa"):
            tensor.validate()

    # --- I6: ρ > 0 ---

    def test_zero_density_raises(self) -> None:
        tensor = SoilTensor(
            uscs_classification="SW",
            liquid_limit=0.0,
            plasticity_index=0.0,
            shear_wave_velocity=300.0,
            void_ratio=0.5,
            is_saturated=False,
            bulk_density_kg_m3=0.0,
        )
        with pytest.raises(LithologicalInputError, match="positiva"):
            tensor.validate()

    # --- I8: Finitud ---

    @pytest.mark.parametrize("field_name,field_value", [
        ("liquid_limit", float("nan")),
        ("plasticity_index", float("inf")),
        ("shear_wave_velocity", float("nan")),
        ("void_ratio", float("-inf")),
        ("bulk_density_kg_m3", float("nan")),
    ])
    def test_non_finite_fields_raise(
        self, field_name: str, field_value: float
    ) -> None:
        """Ningún campo numérico puede ser NaN o ±∞."""
        kwargs = {
            "uscs_classification": "SW",
            "liquid_limit": 0.0,
            "plasticity_index": 0.0,
            "shear_wave_velocity": 300.0,
            "void_ratio": 0.5,
            "is_saturated": False,
            "bulk_density_kg_m3": 1800.0,
        }
        kwargs[field_name] = field_value
        tensor = SoilTensor(**kwargs)
        with pytest.raises(LithologicalInputError, match="finito"):
            tensor.validate()

    # --- I9: Rangos admisibles ---

    def test_ll_exceeds_max_raises(self) -> None:
        tensor = SoilTensor(
            uscs_classification="CH",
            liquid_limit=600.0,
            plasticity_index=100.0,
            shear_wave_velocity=250.0,
            void_ratio=0.6,
            is_saturated=False,
        )
        with pytest.raises(LithologicalInputError, match="máximo"):
            tensor.validate()

    def test_vs_exceeds_max_raises(self) -> None:
        tensor = SoilTensor(
            uscs_classification="SW",
            liquid_limit=0.0,
            plasticity_index=0.0,
            shear_wave_velocity=6000.0,
            void_ratio=0.5,
            is_saturated=False,
        )
        with pytest.raises(LithologicalInputError, match="máximo"):
            tensor.validate()

    def test_void_ratio_exceeds_max_raises(self) -> None:
        tensor = SoilTensor(
            uscs_classification="CH",
            liquid_limit=200.0,
            plasticity_index=100.0,
            shear_wave_velocity=100.0,
            void_ratio=20.0,
            is_saturated=False,
        )
        with pytest.raises(LithologicalInputError, match="máximo"):
            tensor.validate()

    def test_density_below_min_raises(self) -> None:
        tensor = SoilTensor(
            uscs_classification="SW",
            liquid_limit=0.0,
            plasticity_index=0.0,
            shear_wave_velocity=300.0,
            void_ratio=0.5,
            is_saturated=False,
            bulk_density_kg_m3=400.0,
        )
        with pytest.raises(LithologicalInputError, match="rango"):
            tensor.validate()

    def test_density_above_max_raises(self) -> None:
        tensor = SoilTensor(
            uscs_classification="SW",
            liquid_limit=0.0,
            plasticity_index=0.0,
            shear_wave_velocity=300.0,
            void_ratio=0.5,
            is_saturated=False,
            bulk_density_kg_m3=4000.0,
        )
        with pytest.raises(LithologicalInputError, match="rango"):
            tensor.validate()


class TestSoilTensorToDict:
    """Pruebas para SoilTensor.to_dict()."""

    def test_roundtrip_fields(self, nominal_sand_tensor: SoilTensor) -> None:
        d = nominal_sand_tensor.to_dict()
        assert d["uscs_classification"] == "SW"
        assert d["liquid_limit"] == 0.0
        assert d["shear_wave_velocity"] == 300.0
        assert d["is_saturated"] is False

    def test_all_keys_present(self, nominal_sand_tensor: SoilTensor) -> None:
        d = nominal_sand_tensor.to_dict()
        expected_keys = {
            "uscs_classification", "liquid_limit", "plasticity_index",
            "shear_wave_velocity", "void_ratio", "is_saturated",
            "bulk_density_kg_m3",
        }
        assert set(d.keys()) == expected_keys


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              §3  MAGNITUDES DERIVADAS                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class TestDynamicRigidity:
    """
    Pruebas para _compute_dynamic_rigidity_pa.

    Propiedad verificada:
        G_max = ρ · Vs² > 0, con dimensiones [Pa].
    """

    def test_known_computation(
        self, manifold: LithologicalManifold, nominal_sand_tensor: SoilTensor
    ) -> None:
        """G_max = ρ · Vs² > 0, con dimensiones [Pa]."""
        calculated_g = manifold._compute_dynamic_rigidity_pa(nominal_sand_tensor)
        expected_g = nominal_sand_tensor.bulk_density_kg_m3 * nominal_sand_tensor.shear_wave_velocity ** 2
        assert math.isclose(calculated_g, expected_g, rel_tol=1e-9)

    def test_always_positive(self, manifold: LithologicalManifold) -> None:
        """G_max > 0 para cualquier entrada válida."""
        tensor = SoilTensor(
            uscs_classification="ML",
            liquid_limit=30.0,
            plasticity_index=5.0,
            shear_wave_velocity=50.0,
            void_ratio=1.0,
            is_saturated=False,
            bulk_density_kg_m3=1200.0,
        )
        assert manifold._compute_dynamic_rigidity_pa(tensor) > 0.0

    def test_monotone_in_vs(self, manifold: LithologicalManifold) -> None:
        """G_max es estrictamente monótona creciente en Vs."""
        results = []
        for vs in [50.0, 100.0, 200.0, 500.0, 1000.0]:
            tensor = SoilTensor(
                uscs_classification="SW",
                liquid_limit=0.0,
                plasticity_index=0.0,
                shear_wave_velocity=vs,
                void_ratio=0.5,
                is_saturated=False,
            )
            results.append(manifold._compute_dynamic_rigidity_pa(tensor))
        for i in range(len(results) - 1):
            assert results[i] < results[i + 1]

    def test_monotone_in_density(
        self, manifold: LithologicalManifold
    ) -> None:
        """G_max es estrictamente monótona creciente en ρ."""
        results = []
        for rho in [1000.0, 1500.0, 2000.0, 2500.0]:
            tensor = SoilTensor(
                uscs_classification="SW",
                liquid_limit=0.0,
                plasticity_index=0.0,
                shear_wave_velocity=300.0,
                void_ratio=0.5,
                is_saturated=False,
                bulk_density_kg_m3=rho,
            )
            results.append(manifold._compute_dynamic_rigidity_pa(tensor))
        for i in range(len(results) - 1):
            assert results[i] < results[i + 1]

    def test_quadratic_scaling_in_vs(
        self, manifold: LithologicalManifold
    ) -> None:
        """Si Vs se duplica, G_max se cuadruplica (ρ constante)."""
        base = SoilTensor(
            uscs_classification="SW",
            liquid_limit=0.0,
            plasticity_index=0.0,
            shear_wave_velocity=200.0,
            void_ratio=0.5,
            is_saturated=False,
        )
        doubled = SoilTensor(
            uscs_classification="SW",
            liquid_limit=0.0,
            plasticity_index=0.0,
            shear_wave_velocity=400.0,
            void_ratio=0.5,
            is_saturated=False,
        )
        g_base = manifold._compute_dynamic_rigidity_pa(base)
        g_doubled = manifold._compute_dynamic_rigidity_pa(doubled)
        assert g_doubled == pytest.approx(4.0 * g_base)


class TestSwellingPotentialIndex:
    """
    Pruebas para _compute_swelling_potential_index.

    Propiedades verificadas:
        P1: I_sw ∈ [0, 1]
        P2: Monótona creciente en LL (PI fijo)
        P3: Monótona creciente en PI (LL fijo)
        P4: I_sw → 0 cuando LL·PI → 0
        P5: I_sw → 1 cuando LL·PI → ∞
    """

    def test_output_in_unit_interval(
        self, manifold: LithologicalManifold
    ) -> None:
        """P1: Para una muestra de entradas, I_sw ∈ [0, 1]."""
        for ll, pi in [(0, 0), (30, 10), (50, 20), (100, 50), (200, 100)]:
            tensor = SoilTensor(
                uscs_classification="CH",
                liquid_limit=float(ll),
                plasticity_index=float(pi),
                shear_wave_velocity=200.0,
                void_ratio=0.6,
                is_saturated=False,
            )
            index = manifold._compute_swelling_potential_index(tensor)
            assert 0.0 <= index <= 1.0, f"LL={ll}, PI={pi} → {index}"

    def test_monotone_in_ll(self, manifold: LithologicalManifold) -> None:
        """P2: Con PI fijo, I_sw crece con LL."""
        results = []
        for ll in [10.0, 30.0, 50.0, 80.0, 120.0]:
            tensor = SoilTensor(
                uscs_classification="CH",
                liquid_limit=ll,
                plasticity_index=20.0,
                shear_wave_velocity=200.0,
                void_ratio=0.6,
                is_saturated=False,
            )
            results.append(manifold._compute_swelling_potential_index(tensor))
        for i in range(len(results) - 1):
            assert results[i] <= results[i + 1]

    def test_monotone_in_pi(self, manifold: LithologicalManifold) -> None:
        """P3: Con LL fijo, I_sw crece con PI."""
        results = []
        for pi in [0.0, 10.0, 25.0, 40.0, 60.0]:
            tensor = SoilTensor(
                uscs_classification="CH",
                liquid_limit=80.0,
                plasticity_index=pi,
                shear_wave_velocity=200.0,
                void_ratio=0.6,
                is_saturated=False,
            )
            results.append(manifold._compute_swelling_potential_index(tensor))
        for i in range(len(results) - 1):
            assert results[i] <= results[i + 1]

    def test_near_zero_for_non_plastic(
        self, manifold: LithologicalManifold
    ) -> None:
        """P4: Arena limpia (LL=PI=0) → I_sw cercano a 0."""
        tensor = SoilTensor(
            uscs_classification="SW",
            liquid_limit=0.0,
            plasticity_index=0.0,
            shear_wave_velocity=300.0,
            void_ratio=0.5,
            is_saturated=False,
        )
        index = manifold._compute_swelling_potential_index(tensor)
        assert index < 0.1

    def test_near_one_for_extreme_clay(
        self, manifold: LithologicalManifold
    ) -> None:
        """P5: Bentonita extrema (LL=300, PI=250) → I_sw cercano a 1."""
        tensor = SoilTensor(
            uscs_classification="CH",
            liquid_limit=300.0,
            plasticity_index=250.0,
            shear_wave_velocity=100.0,
            void_ratio=2.0,
            is_saturated=False,
        )
        index = manifold._compute_swelling_potential_index(tensor)
        assert index > 0.95


class TestYieldingSusceptibilityIndex:
    """
    Pruebas para _compute_yielding_susceptibility_index.

    Propiedades verificadas:
        P1: I_y ∈ [0, 1]
        P2: Monótona creciente en e₀
        P3: Monótona decreciente en Vs
        P4: Adimensional (por normalización)
    """

    def test_output_in_unit_interval(
        self, manifold: LithologicalManifold
    ) -> None:
        for e0, vs in [(0.3, 500), (0.8, 200), (1.5, 80), (3.0, 50)]:
            tensor = SoilTensor(
                uscs_classification="CL",
                liquid_limit=40.0,
                plasticity_index=15.0,
                shear_wave_velocity=float(vs),
                void_ratio=float(e0),
                is_saturated=False,
            )
            index = manifold._compute_yielding_susceptibility_index(tensor)
            assert 0.0 <= index <= 1.0, f"e₀={e0}, Vs={vs} → {index}"

    def test_monotone_increasing_in_void_ratio(
        self, manifold: LithologicalManifold
    ) -> None:
        """P2: I_y crece con e₀ (Vs fijo)."""
        results = []
        for e0 in [0.3, 0.5, 0.7, 0.9, 1.2]:
            tensor = SoilTensor(
                uscs_classification="CL",
                liquid_limit=40.0,
                plasticity_index=15.0,
                shear_wave_velocity=200.0,
                void_ratio=e0,
                is_saturated=False,
            )
            results.append(
                manifold._compute_yielding_susceptibility_index(tensor)
            )
        for i in range(len(results) - 1):
            assert results[i] <= results[i + 1]

    def test_monotone_decreasing_in_vs(
        self, manifold: LithologicalManifold
    ) -> None:
        """P3: I_y decrece con Vs (e₀ fijo)."""
        results = []
        for vs in [50.0, 100.0, 200.0, 400.0, 800.0]:
            tensor = SoilTensor(
                uscs_classification="CL",
                liquid_limit=40.0,
                plasticity_index=15.0,
                shear_wave_velocity=vs,
                void_ratio=0.8,
                is_saturated=False,
            )
            results.append(
                manifold._compute_yielding_susceptibility_index(tensor)
            )
        for i in range(len(results) - 1):
            assert results[i] >= results[i + 1]

    def test_zero_void_ratio_gives_zero(
        self, manifold: LithologicalManifold
    ) -> None:
        """Con e₀ = 0, el índice debe ser 0."""
        tensor = SoilTensor(
            uscs_classification="SW",
            liquid_limit=0.0,
            plasticity_index=0.0,
            shear_wave_velocity=300.0,
            void_ratio=0.0,
            is_saturated=False,
        )
        assert manifold._compute_yielding_susceptibility_index(tensor) == 0.0

    def test_saturates_at_one(
        self, manifold: LithologicalManifold
    ) -> None:
        """Para e₀ muy alto y Vs muy bajo, I_y satura en 1.0."""
        tensor = SoilTensor(
            uscs_classification="CH",
            liquid_limit=200.0,
            plasticity_index=100.0,
            shear_wave_velocity=10.0,
            void_ratio=10.0,
            is_saturated=False,
            bulk_density_kg_m3=1050.0,
        )
        index = manifold._compute_yielding_susceptibility_index(tensor)
        assert index == pytest.approx(1.0)


class TestCompressionIndexProxy:
    """
    Pruebas para _compute_compression_index_proxy (Skempton).

    Correlación: Cc = 0.009 · (LL - 10), con Cc ≥ 0.
    """

    def test_known_value(self, manifold: LithologicalManifold) -> None:
        """Cc(LL=50) = 0.009 · 40 = 0.36."""
        tensor = SoilTensor(
            uscs_classification="CL",
            liquid_limit=50.0,
            plasticity_index=20.0,
            shear_wave_velocity=200.0,
            void_ratio=0.8,
            is_saturated=False,
        )
        cc = manifold._compute_compression_index_proxy(tensor)
        assert cc == pytest.approx(0.009 * 40.0)

    def test_zero_for_low_ll(self, manifold: LithologicalManifold) -> None:
        """Cc = 0 cuando LL ≤ 10."""
        tensor = SoilTensor(
            uscs_classification="SW",
            liquid_limit=5.0,
            plasticity_index=0.0,
            shear_wave_velocity=300.0,
            void_ratio=0.5,
            is_saturated=False,
        )
        assert manifold._compute_compression_index_proxy(tensor) == 0.0

    def test_nonnegative(self, manifold: LithologicalManifold) -> None:
        """Cc ≥ 0 siempre."""
        tensor = SoilTensor(
            uscs_classification="SW",
            liquid_limit=0.0,
            plasticity_index=0.0,
            shear_wave_velocity=300.0,
            void_ratio=0.5,
            is_saturated=False,
        )
        assert manifold._compute_compression_index_proxy(tensor) >= 0.0

    def test_monotone_in_ll(self, manifold: LithologicalManifold) -> None:
        """Cc crece monótonamente con LL."""
        results = []
        for ll in [0.0, 10.0, 30.0, 60.0, 100.0]:
            tensor = SoilTensor(
                uscs_classification="CH",
                liquid_limit=ll,
                plasticity_index=min(ll, 10.0),
                shear_wave_velocity=200.0,
                void_ratio=0.8,
                is_saturated=False,
            )
            results.append(manifold._compute_compression_index_proxy(tensor))
        for i in range(len(results) - 1):
            assert results[i] <= results[i + 1]


class TestLiquefactionSusceptibilityIndex:
    """
    Pruebas para _compute_liquefaction_susceptibility_index.

    Propiedades verificadas:
        P1: I_liq ∈ [0, 1]
        P2: I_liq = 0 si no saturado
        P3: I_liq = 0 si no arenoso
        P4: I_liq = 0 si Vs ≥ Vs_crit
        P5: I_liq crece cuando Vs decrece (Vs < Vs_crit)
        P6: I_liq es no lineal (exponente n = 2)
    """

    def test_zero_when_unsaturated(
        self, manifold: LithologicalManifold
    ) -> None:
        """P2: Arena seca no tiene susceptibilidad a licuación."""
        tensor = SoilTensor(
            uscs_classification="SP",
            liquid_limit=0.0,
            plasticity_index=0.0,
            shear_wave_velocity=100.0,
            void_ratio=0.7,
            is_saturated=False,
        )
        assert manifold._compute_liquefaction_susceptibility_index(tensor) == 0.0
        assert math.isclose(manifold._compute_liquefaction_susceptibility_index(tensor), 0.0, abs_tol=1e-12)

    def test_zero_when_not_sand(
        self, manifold: LithologicalManifold
    ) -> None:
        """P3: Arcilla saturada no produce licuación en este modelo."""
        tensor = SoilTensor(
            uscs_classification="CH",
            liquid_limit=60.0,
            plasticity_index=30.0,
            shear_wave_velocity=100.0,
            void_ratio=0.8,
            is_saturated=True,
        )
        assert manifold._compute_liquefaction_susceptibility_index(tensor) == 0.0

    def test_zero_when_vs_at_critical(
        self, manifold: LithologicalManifold
    ) -> None:
        """P4: Vs = Vs_crit → I_liq = 0."""
        tensor = SoilTensor(
            uscs_classification="SP",
            liquid_limit=0.0,
            plasticity_index=0.0,
            shear_wave_velocity=C.CRITICAL_LIQUEFACTION_VS_M_S,
            void_ratio=0.7,
            is_saturated=True,
        )
        assert manifold._compute_liquefaction_susceptibility_index(tensor) == 0.0

    def test_zero_when_vs_above_critical(
        self, manifold: LithologicalManifold
    ) -> None:
        """P4: Vs > Vs_crit → I_liq = 0."""
        tensor = SoilTensor(
            uscs_classification="SP",
            liquid_limit=0.0,
            plasticity_index=0.0,
            shear_wave_velocity=200.0,
            void_ratio=0.7,
            is_saturated=True,
        )
        assert manifold._compute_liquefaction_susceptibility_index(tensor) == 0.0

    def test_positive_when_vs_below_critical(
        self, manifold: LithologicalManifold
    ) -> None:
        """Arena saturada con Vs < Vs_crit produce I_liq > 0."""
        tensor = SoilTensor(
            uscs_classification="SP",
            liquid_limit=0.0,
            plasticity_index=0.0,
            shear_wave_velocity=100.0,
            void_ratio=0.7,
            is_saturated=True,
        )
        index = manifold._compute_liquefaction_susceptibility_index(tensor)
        assert 0.0 < index <= 1.0

    def test_monotone_decreasing_in_vs(
        self, manifold: LithologicalManifold
    ) -> None:
        """P5: I_liq crece cuando Vs decrece (por debajo de Vs_crit)."""
        vs_values = [140.0, 120.0, 100.0, 80.0, 50.0, 10.0]
        results = []
        for vs in vs_values:
            tensor = SoilTensor(
                uscs_classification="SP",
                liquid_limit=0.0,
                plasticity_index=0.0,
                shear_wave_velocity=vs,
                void_ratio=0.7,
                is_saturated=True,
            )
            results.append(
                manifold._compute_liquefaction_susceptibility_index(tensor)
            )
        for i in range(len(results) - 1):
            assert results[i] <= results[i + 1]

    def test_nonlinear_scaling(
        self, manifold: LithologicalManifold
    ) -> None:
        """
        P6: Con exponente n=2, la relación no es lineal.

        Para Vs=75 (mitad del rango [0, 150]):
            r = (150 - 75)/150 = 0.5
            I_liq = 0.5² = 0.25  (no 0.5 que sería lineal)
        """
        tensor = SoilTensor(
            uscs_classification="SP",
            liquid_limit=0.0,
            plasticity_index=0.0,
            shear_wave_velocity=75.0,
            void_ratio=0.7,
            is_saturated=True,
        )
        index = manifold._compute_liquefaction_susceptibility_index(tensor)
        assert index == pytest.approx(0.25, abs=1e-6)

    def test_maximum_at_vs_near_zero(
        self, manifold: LithologicalManifold
    ) -> None:
        """I_liq → 1 cuando Vs → 0."""
        tensor = SoilTensor(
            uscs_classification="SP",
            liquid_limit=0.0,
            plasticity_index=0.0,
            shear_wave_velocity=0.1,
            void_ratio=0.7,
            is_saturated=True,
        )
        index = manifold._compute_liquefaction_susceptibility_index(tensor)
        assert index == pytest.approx(1.0, abs=0.01)

    def test_in_unit_interval_exhaustive(
        self, manifold: LithologicalManifold
    ) -> None:
        """P1: I_liq ∈ [0, 1] para barrido de Vs."""
        for vs in [0.1, 10, 50, 100, 149, 150, 200, 500]:
            tensor = SoilTensor(
                uscs_classification="SP",
                liquid_limit=0.0,
                plasticity_index=0.0,
                shear_wave_velocity=float(vs),
                void_ratio=0.7,
                is_saturated=True,
            )
            index = manifold._compute_liquefaction_susceptibility_index(tensor)
            assert 0.0 <= index <= 1.0, f"Vs={vs} → {index}"

    def test_dual_sand_group_triggers(
        self, manifold: LithologicalManifold
    ) -> None:
        """Grupo dual SW-SM debe ser reconocido como arenoso."""
        tensor = SoilTensor(
            uscs_classification="SW-SM",
            liquid_limit=20.0,
            plasticity_index=5.0,
            shear_wave_velocity=100.0,
            void_ratio=0.6,
            is_saturated=True,
        )
        index = manifold._compute_liquefaction_susceptibility_index(tensor)
        assert index > 0.0


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              §4  REGLAS DIAGNÓSTICAS Y CARTUCHOS                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class TestDiagnosticRulesSwelling:
    """Pruebas de la regla de expansividad."""

    def test_swelling_cartridge_emitted(
        self,
        manifold: LithologicalManifold,
        expansive_clay_tensor: SoilTensor,
    ) -> None:
        """Arcilla expansiva debe emitir cartucho SwellingPlasmon."""
        report = manifold._evaluate_diagnostic_rules(expansive_clay_tensor)
        swelling_cartridges = [
            c for c in report.emitted_cartridges
            if isinstance(c, SwellingPlasmonCartridge)
        ]
        assert len(swelling_cartridges) == 1

    def test_swelling_rule_activated(
        self,
        manifold: LithologicalManifold,
        expansive_clay_tensor: SoilTensor,
    ) -> None:
        report = manifold._evaluate_diagnostic_rules(expansive_clay_tensor)
        assert DiagnosticRule.SWELLING_POTENTIAL_DETECTED in report.activated_rules

    def test_no_swelling_for_sand(
        self,
        manifold: LithologicalManifold,
        nominal_sand_tensor: SoilTensor,
    ) -> None:
        """Arena limpia no debe activar regla de expansividad."""
        report = manifold._evaluate_diagnostic_rules(nominal_sand_tensor)
        assert DiagnosticRule.SWELLING_POTENTIAL_DETECTED not in report.activated_rules

    def test_swelling_boundary_below_threshold(
        self, manifold: LithologicalManifold
    ) -> None:
        """Justo debajo de umbrales LL y PI: no debe activarse."""
        tensor = SoilTensor(
            uscs_classification="CL",
            liquid_limit=49.9,
            plasticity_index=19.9,
            shear_wave_velocity=200.0,
            void_ratio=0.6,
            is_saturated=False,
        )
        report = manifold._evaluate_diagnostic_rules(tensor)
        assert DiagnosticRule.SWELLING_POTENTIAL_DETECTED not in report.activated_rules

    def test_swelling_boundary_at_threshold(
        self, manifold: LithologicalManifold
    ) -> None:
        """Exactamente en umbrales LL y PI: debe activarse."""
        tensor = SoilTensor(
            uscs_classification="CL",
            liquid_limit=50.0,
            plasticity_index=20.0,
            shear_wave_velocity=200.0,
            void_ratio=0.6,
            is_saturated=False,
        )
        report = manifold._evaluate_diagnostic_rules(tensor)
        assert DiagnosticRule.SWELLING_POTENTIAL_DETECTED in report.activated_rules

    def test_swelling_only_ll_above_not_pi(
        self, manifold: LithologicalManifold
    ) -> None:
        """Si solo LL supera umbral pero PI no, la regla no se activa."""
        tensor = SoilTensor(
            uscs_classification="ML",
            liquid_limit=60.0,
            plasticity_index=10.0,
            shear_wave_velocity=200.0,
            void_ratio=0.6,
            is_saturated=False,
        )
        report = manifold._evaluate_diagnostic_rules(tensor)
        assert DiagnosticRule.SWELLING_POTENTIAL_DETECTED not in report.activated_rules


class TestDiagnosticRulesYielding:
    """Pruebas de la regla de cedencia/compresibilidad."""

    def test_yielding_cartridge_emitted(
        self,
        manifold: LithologicalManifold,
        soft_clay_tensor: SoilTensor,
    ) -> None:
        report = manifold._evaluate_diagnostic_rules(soft_clay_tensor)
        yielding_cartridges = [
            c for c in report.emitted_cartridges
            if isinstance(c, YieldingPhononCartridge)
        ]
        assert len(yielding_cartridges) == 1

    def test_yielding_rule_activated(
        self,
        manifold: LithologicalManifold,
        soft_clay_tensor: SoilTensor,
    ) -> None:
        report = manifold._evaluate_diagnostic_rules(soft_clay_tensor)
        assert DiagnosticRule.YIELDING_SUSCEPTIBILITY_DETECTED in report.activated_rules

    def test_no_yielding_for_stiff_soil(
        self,
        manifold: LithologicalManifold,
        nominal_sand_tensor: SoilTensor,
    ) -> None:
        """Arena con e₀=0.5 y Vs=300: no debe activar cedencia."""
        report = manifold._evaluate_diagnostic_rules(nominal_sand_tensor)
        assert DiagnosticRule.YIELDING_SUSCEPTIBILITY_DETECTED not in report.activated_rules

    def test_yielding_requires_both_conditions(
        self, manifold: LithologicalManifold
    ) -> None:
        """Alto void ratio pero alta Vs: no debe activar."""
        tensor = SoilTensor(
            uscs_classification="CL",
            liquid_limit=40.0,
            plasticity_index=15.0,
            shear_wave_velocity=250.0,
            void_ratio=1.0,
            is_saturated=False,
        )
        report = manifold._evaluate_diagnostic_rules(tensor)
        assert DiagnosticRule.YIELDING_SUSCEPTIBILITY_DETECTED not in report.activated_rules

    def test_yielding_cartridge_has_skempton_cc(
        self,
        manifold: LithologicalManifold,
        soft_clay_tensor: SoilTensor,
    ) -> None:
        """El cartucho debe contener Cc según Skempton."""
        report = manifold._evaluate_diagnostic_rules(soft_clay_tensor)
        cart = [
            c for c in report.emitted_cartridges
            if isinstance(c, YieldingPhononCartridge)
        ][0]
        expected_cc = C.SKEMPTON_CC_FACTOR * (
            soft_clay_tensor.liquid_limit - C.SKEMPTON_CC_OFFSET
        )
        assert cart.compression_index == pytest.approx(expected_cc)


class TestDiagnosticRulesLiquefaction:
    """Pruebas de la regla de licuación."""

    def test_liquefaction_cartridge_emitted(
        self,
        manifold: LithologicalManifold,
        liquefiable_sand_tensor: SoilTensor,
    ) -> None:
        report = manifold._evaluate_diagnostic_rules(liquefiable_sand_tensor)
        liq_carts = [
            c for c in report.emitted_cartridges
            if isinstance(c, LiquefactionSolitonCartridge)
        ]
        assert len(liq_carts) == 1

    def test_liquefaction_rule_activated(
        self,
        manifold: LithologicalManifold,
        liquefiable_sand_tensor: SoilTensor,
    ) -> None:
        report = manifold._evaluate_diagnostic_rules(liquefiable_sand_tensor)
        assert DiagnosticRule.LIQUEFACTION_SUSCEPTIBILITY_DETECTED in report.activated_rules

    def test_no_liquefaction_for_dry_sand(
        self, manifold: LithologicalManifold
    ) -> None:
        tensor = SoilTensor(
            uscs_classification="SP",
            liquid_limit=0.0,
            plasticity_index=0.0,
            shear_wave_velocity=100.0,
            void_ratio=0.7,
            is_saturated=False,
        )
        report = manifold._evaluate_diagnostic_rules(tensor)
        assert DiagnosticRule.LIQUEFACTION_SUSCEPTIBILITY_DETECTED not in report.activated_rules

    def test_liquefaction_cartridge_uses_correct_field_name(
        self,
        manifold: LithologicalManifold,
        liquefiable_sand_tensor: SoilTensor,
    ) -> None:
        """Verifica que el campo se llama 'susceptibility_index', no 'cyclic_stress_ratio'."""
        report = manifold._evaluate_diagnostic_rules(liquefiable_sand_tensor)
        cart = [
            c for c in report.emitted_cartridges
            if isinstance(c, LiquefactionSolitonCartridge)
        ][0]
        assert hasattr(cart, "susceptibility_index")
        assert not hasattr(cart, "cyclic_stress_ratio")


class TestDiagnosticRulesLowRigidity:
    """Pruebas de la regla de rigidez dinámica baja."""

    def test_low_rigidity_detected(
        self, manifold: LithologicalManifold
    ) -> None:
        """Suelo blando con G_max < 25 MPa."""
        tensor = SoilTensor(
            uscs_classification="ML",
            liquid_limit=35.0,
            plasticity_index=10.0,
            shear_wave_velocity=80.0,
            void_ratio=0.6,
            is_saturated=False,
            bulk_density_kg_m3=1500.0,
        )
        # G_max = 1500 * 80^2 = 9,600,000 Pa < 25e6
        report = manifold._evaluate_diagnostic_rules(tensor)
        assert DiagnosticRule.LOW_DYNAMIC_RIGIDITY_REGIME in report.activated_rules

    def test_high_rigidity_not_flagged(
        self,
        manifold: LithologicalManifold,
        nominal_sand_tensor: SoilTensor,
    ) -> None:
        """G_max = 1800 · 300² = 162 MPa >> 25 MPa."""
        report = manifold._evaluate_diagnostic_rules(nominal_sand_tensor)
        assert DiagnosticRule.LOW_DYNAMIC_RIGIDITY_REGIME not in report.activated_rules


class TestMultipleCartridgesCoexistence:
    """Pruebas de que múltiples reglas pueden coexistir."""

    def test_swelling_and_yielding_coexist(
        self, manifold: LithologicalManifold
    ) -> None:
        """Arcilla expansiva y blanda activa ambas reglas."""
        tensor = SoilTensor(
            uscs_classification="CH",
            liquid_limit=70.0,
            plasticity_index=40.0,
            shear_wave_velocity=100.0,
            void_ratio=1.2,
            is_saturated=False,
            bulk_density_kg_m3=1500.0,
        )
        report = manifold._evaluate_diagnostic_rules(tensor)
        assert DiagnosticRule.SWELLING_POTENTIAL_DETECTED in report.activated_rules
        assert DiagnosticRule.YIELDING_SUSCEPTIBILITY_DETECTED in report.activated_rules
        assert len(report.emitted_cartridges) >= 2

    def test_no_rules_for_competent_soil(
        self, manifold: LithologicalManifold
    ) -> None:
        """Suelo competente no activa ninguna regla de patología."""
        tensor = SoilTensor(
            uscs_classification="GW",
            liquid_limit=0.0,
            plasticity_index=0.0,
            shear_wave_velocity=400.0,
            void_ratio=0.3,
            is_saturated=False,
        )
        report = manifold._evaluate_diagnostic_rules(tensor)
        pathological_rules = {
            DiagnosticRule.SWELLING_POTENTIAL_DETECTED,
            DiagnosticRule.YIELDING_SUSCEPTIBILITY_DETECTED,
            DiagnosticRule.LIQUEFACTION_SUSCEPTIBILITY_DETECTED,
        }
        assert not pathological_rules.intersection(report.activated_rules)
        assert len(report.emitted_cartridges) == 0


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              §5  SINGULARIDADES LITOLÓGICAS                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class TestPeatSingularity:
    """Pruebas de detección de turba como singularidad."""

    def test_peat_raises_singularity(
        self,
        manifold: LithologicalManifold,
        peat_tensor: SoilTensor,
    ) -> None:
        with pytest.raises(LithologicalSingularityError, match="Turba"):
            manifold._evaluate_diagnostic_rules(peat_tensor)

    def test_peat_in_full_call_raises(
        self,
        manifold: LithologicalManifold,
    ) -> None:
        payload = _make_payload(uscs="PT", liquid_limit=200.0,
                                plasticity_index=100.0, vs=60.0,
                                void_ratio=8.0, is_saturated=True,
                                bulk_density_kg_m3=1050.0)
        with pytest.raises(LithologicalSingularityError):
            manifold(payload, {})

    def test_peat_case_insensitive(
        self, manifold: LithologicalManifold
    ) -> None:
        """'pt', 'Pt', 'pT' deben detectarse como turba."""
        for uscs_variant in ["pt", "Pt", "pT", " PT "]:
            tensor = SoilTensor(
                uscs_classification=uscs_variant,
                liquid_limit=200.0,
                plasticity_index=100.0,
                shear_wave_velocity=60.0,
                void_ratio=8.0,
                is_saturated=True,
                bulk_density_kg_m3=1050.0,
            )
            with pytest.raises(LithologicalSingularityError):
                manifold._evaluate_diagnostic_rules(tensor)

    def test_yielding_susceptibility_approaches_one(
        self, manifold: LithologicalManifold, peat_tensor: SoilTensor
    ) -> None:
        """La turba (PT) actúa como una singularidad y el índice de cedencia se aproxima asintóticamente a 1.0."""
        yielding_susceptibility_index = manifold._compute_yielding_susceptibility_index(peat_tensor)
        assert math.isclose(yielding_susceptibility_index, 1.0, rel_tol=1e-9)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              §6  MORFISMO COMPLETO (__call__) — INTEGRACIÓN                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class TestMorphismIntegration:
    """Pruebas de integración del operador completo."""

    def test_nominal_call_returns_categorical_state(
        self,
        manifold: LithologicalManifold,
        nominal_payload: Dict[str, Any],
        empty_context: Dict[str, Any],
    ) -> None:
        state = manifold(nominal_payload, empty_context)
        assert hasattr(state, "payload")
        assert hasattr(state, "context")
        assert hasattr(state, "validated_strata")

    def test_validated_strata_contains_physics(
        self,
        manifold: LithologicalManifold,
        nominal_payload: Dict[str, Any],
        empty_context: Dict[str, Any],
    ) -> None:
        import sys
        state = manifold(nominal_payload, empty_context)
        # Using Stratum.PHYSICS which was mocked as 4
        assert sys.modules["app.core.schemas"].Stratum.PHYSICS in state.validated_strata

    def test_payload_enriched_with_derived_quantities(
        self,
        manifold: LithologicalManifold,
        nominal_payload: Dict[str, Any],
        empty_context: Dict[str, Any],
    ) -> None:
        state = manifold(nominal_payload, empty_context)
        enriched_keys = {
            "dynamic_rigidity_modulus_pa",
            "swelling_potential_index",
            "yielding_susceptibility_index",
            "liquefaction_susceptibility_index",
            "lithology_activated_rules",
        }
        assert enriched_keys.issubset(set(state.payload.keys()))

    def test_context_contains_lithology_report(
        self,
        manifold: LithologicalManifold,
        nominal_payload: Dict[str, Any],
        empty_context: Dict[str, Any],
    ) -> None:
        state = manifold(nominal_payload, empty_context)
        assert "lithology_report" in state.context
        report = state.context["lithology_report"]
        assert "soil_tensor" in report
        assert "dynamic_rigidity_modulus_pa" in report
        assert "activated_rules" in report

    def test_context_contains_synaptic_registry(
        self,
        manifold: LithologicalManifold,
        nominal_payload: Dict[str, Any],
        empty_context: Dict[str, Any],
    ) -> None:
        state = manifold(nominal_payload, empty_context)
        assert "synaptic_registry" in state.context

    def test_original_payload_not_mutated(
        self, manifold: LithologicalManifold
    ) -> None:
        """El payload original no debe ser modificado (copia defensiva)."""
        payload = _make_payload()
        original_keys = set(payload.keys())
        original_values = dict(payload)
        manifold(payload, {})
        assert set(payload.keys()) == original_keys
        assert payload == original_values

    def test_original_context_not_mutated(
        self, manifold: LithologicalManifold
    ) -> None:
        """El contexto original no debe ser modificado."""
        context: Dict[str, Any] = {"existing_key": "existing_value"}
        original = dict(context)
        manifold(_make_payload(), context)
        assert context == original

    def test_synaptic_registry_appends_not_overwrites(
        self, manifold: LithologicalManifold
    ) -> None:
        """Si el contexto ya tiene cartuchos, se agregan, no se reemplazan."""
        existing_cartridge = MagicMock()
        context: Dict[str, Any] = {
            "synaptic_registry": [existing_cartridge],
        }
        payload = _make_payload(
            uscs="CH", liquid_limit=70.0, plasticity_index=40.0,
            vs=180.0, void_ratio=0.9, is_saturated=True,
            bulk_density_kg_m3=1650.0,
        )
        state = manifold(payload, context)
        registry = state.context["synaptic_registry"]
        assert existing_cartridge in registry
        assert len(registry) > 1

    def test_default_values_when_keys_missing(
        self, manifold: LithologicalManifold
    ) -> None:
        """Payload mínimo con solo USCS debe funcionar con defaults."""
        payload: Dict[str, Any] = {"uscs": "SW"}
        state = manifold(payload, {})
        assert state.payload["dynamic_rigidity_modulus_pa"] > 0

    def test_input_error_propagates(
        self, manifold: LithologicalManifold
    ) -> None:
        """Datos inválidos deben propagar LithologicalInputError."""
        payload = _make_payload(vs=-100.0)
        with pytest.raises(LithologicalInputError):
            manifold(payload, {})

    def test_type_error_wraps_in_input_error(
        self, manifold: LithologicalManifold
    ) -> None:
        """Tipo incompatible se envuelve en LithologicalInputError."""
        payload = _make_payload(vs="not_a_number")
        with pytest.raises(LithologicalInputError, match="Degeneración"):
            manifold(payload, {})

    def test_activated_rules_are_strings_in_output(
        self, manifold: LithologicalManifold
    ) -> None:
        """Las reglas activadas en el payload deben ser strings (enum.value)."""
        payload = _make_payload(
            uscs="CH", liquid_limit=70.0, plasticity_index=40.0,
            vs=180.0, void_ratio=0.9, is_saturated=True,
            bulk_density_kg_m3=1650.0,
        )
        state = manifold(payload, {})
        rules = state.payload["lithology_activated_rules"]
        assert all(isinstance(r, str) for r in rules)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              §7  PROPIEDADES DE NORMALIZACIÓN [0, 1]                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class TestNormalizationInvariants:
    """
    Verificación exhaustiva de que todos los índices de susceptibilidad
    permanecen en [0, 1] bajo un barrido amplio de parámetros.

    Esta clase implementa un test de tipo property-based simplificado,
    barriendo el espacio de parámetros en una malla discreta.
    """

    USCS_SAMPLES = ["SW", "SP", "GW", "CL", "CH", "ML", "MH", "OH"]
    LL_SAMPLES = [0.0, 25.0, 50.0, 100.0, 200.0]
    VS_SAMPLES = [10.0, 80.0, 150.0, 300.0, 1000.0]
    E0_SAMPLES = [0.0, 0.4, 0.8, 1.5, 5.0]
    SAT_SAMPLES = [True, False]

    @pytest.mark.parametrize("uscs", USCS_SAMPLES)
    def test_indices_bounded_across_uscs(
        self, manifold: LithologicalManifold, uscs: str
    ) -> None:
        for ll in self.LL_SAMPLES:
            pi = min(ll, 30.0)
            for vs in self.VS_SAMPLES:
                for e0 in self.E0_SAMPLES:
                    for sat in self.SAT_SAMPLES:
                        tensor = SoilTensor(
                            uscs_classification=uscs,
                            liquid_limit=ll,
                            plasticity_index=pi,
                            shear_wave_velocity=vs,
                            void_ratio=e0,
                            is_saturated=sat,
                        )
                        sw = manifold._compute_swelling_potential_index(tensor)
                        yl = manifold._compute_yielding_susceptibility_index(tensor)
                        lq = manifold._compute_liquefaction_susceptibility_index(tensor)

                        assert 0.0 <= sw <= 1.0, (
                            f"I_sw={sw} fuera de [0,1] para {uscs}, "
                            f"LL={ll}, PI={pi}, Vs={vs}, e₀={e0}"
                        )
                        assert 0.0 <= yl <= 1.0, (
                            f"I_y={yl} fuera de [0,1] para {uscs}, "
                            f"LL={ll}, PI={pi}, Vs={vs}, e₀={e0}"
                        )
                        assert 0.0 <= lq <= 1.0, (
                            f"I_liq={lq} fuera de [0,1] para {uscs}, "
                            f"LL={ll}, PI={pi}, Vs={vs}, e₀={e0}"
                        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              §8  ROBUSTEZ NUMÉRICA Y CASOS EXTREMOS                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class TestNumericalRobustness:
    """Pruebas de estabilidad numérica bajo condiciones extremas."""

    def test_very_small_vs(self, manifold: LithologicalManifold) -> None:
        """Vs muy pequeño (pero > 0) no debe producir NaN ni overflow."""
        tensor = SoilTensor(
            uscs_classification="ML",
            liquid_limit=30.0,
            plasticity_index=10.0,
            shear_wave_velocity=0.01,
            void_ratio=0.5,
            is_saturated=False,
            bulk_density_kg_m3=1500.0,
        )
        g_max = manifold._compute_dynamic_rigidity_pa(tensor)
        assert math.isfinite(g_max)
        assert g_max > 0.0

    def test_very_large_void_ratio(
        self, manifold: LithologicalManifold
    ) -> None:
        """e₀ = 14 (máximo admisible) no produce overflow."""
        tensor = SoilTensor(
            uscs_classification="CH",
            liquid_limit=300.0,
            plasticity_index=150.0,
            shear_wave_velocity=50.0,
            void_ratio=14.0,
            is_saturated=False,
            bulk_density_kg_m3=1050.0,
        )
        index = manifold._compute_yielding_susceptibility_index(tensor)
        assert math.isfinite(index)
        assert 0.0 <= index <= 1.0

    def test_maximum_admissible_ll_and_pi(
        self, manifold: LithologicalManifold
    ) -> None:
        """LL y PI en sus máximos admisibles."""
        tensor = SoilTensor(
            uscs_classification="CH",
            liquid_limit=C.LIQUID_LIMIT_MAX,
            plasticity_index=C.PLASTICITY_INDEX_MAX,
            shear_wave_velocity=50.0,
            void_ratio=5.0,
            is_saturated=False,
            bulk_density_kg_m3=1050.0,
        )
        sw = manifold._compute_swelling_potential_index(tensor)
        assert math.isfinite(sw)
        assert 0.0 <= sw <= 1.0

    def test_all_minimum_valid_values(
        self, manifold: LithologicalManifold
    ) -> None:
        """Todos los valores en sus mínimos válidos."""
        tensor = SoilTensor(
            uscs_classification="SW",
            liquid_limit=0.0,
            plasticity_index=0.0,
            shear_wave_velocity=0.01,
            void_ratio=0.0,
            is_saturated=False,
            bulk_density_kg_m3=C.BULK_DENSITY_MIN,
        )
        report = manifold._evaluate_diagnostic_rules(tensor)
        assert math.isfinite(report.dynamic_rigidity_modulus_pa)
        assert math.isfinite(report.swelling_potential_index)
        assert math.isfinite(report.yielding_susceptibility_index)
        assert math.isfinite(report.liquefaction_susceptibility_index)

    def test_g_max_precision_for_large_vs(
        self, manifold: LithologicalManifold
    ) -> None:
        """Vs grande no pierde precisión por overflow de Vs²."""
        tensor = SoilTensor(
            uscs_classification="GW",
            liquid_limit=0.0,
            plasticity_index=0.0,
            shear_wave_velocity=4000.0,
            void_ratio=0.3,
            is_saturated=False,
            bulk_density_kg_m3=2500.0,
        )
        g_max = manifold._compute_dynamic_rigidity_pa(tensor)
        expected = 2500.0 * 4000.0 * 4000.0
        assert g_max == pytest.approx(expected)
        assert math.isfinite(g_max)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              §9  SERIALIZACIÓN Y TRAZABILIDAD                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class TestDiagnosticReportSerialization:
    """Pruebas de LithologyDiagnosticReport.to_dict()."""

    def test_report_to_dict_contains_all_keys(
        self,
        manifold: LithologicalManifold,
        expansive_clay_tensor: SoilTensor,
    ) -> None:
        report = manifold._evaluate_diagnostic_rules(expansive_clay_tensor)
        d = report.to_dict()
        expected_keys = {
            "soil_tensor",
            "dynamic_rigidity_modulus_pa",
            "swelling_potential_index",
            "yielding_susceptibility_index",
            "liquefaction_susceptibility_index",
            "activated_rules",
            "recommendations",
            "emitted_cartridges_count",
        }
        assert set(d.keys()) == expected_keys

    def test_activated_rules_serialized_as_strings(
        self,
        manifold: LithologicalManifold,
        expansive_clay_tensor: SoilTensor,
    ) -> None:
        report = manifold._evaluate_diagnostic_rules(expansive_clay_tensor)
        d = report.to_dict()
        assert all(isinstance(r, str) for r in d["activated_rules"])

    def test_soil_tensor_nested_dict(
        self,
        manifold: LithologicalManifold,
        expansive_clay_tensor: SoilTensor,
    ) -> None:
        report = manifold._evaluate_diagnostic_rules(expansive_clay_tensor)
        d = report.to_dict()
        assert isinstance(d["soil_tensor"], dict)
        assert "uscs_classification" in d["soil_tensor"]

    def test_cartridge_count_matches(
        self,
        manifold: LithologicalManifold,
        expansive_clay_tensor: SoilTensor,
    ) -> None:
        report = manifold._evaluate_diagnostic_rules(expansive_clay_tensor)
        d = report.to_dict()
        assert d["emitted_cartridges_count"] == len(report.emitted_cartridges)

    def test_recommendations_is_list(
        self,
        manifold: LithologicalManifold,
        nominal_sand_tensor: SoilTensor,
    ) -> None:
        report = manifold._evaluate_diagnostic_rules(nominal_sand_tensor)
        d = report.to_dict()
        assert isinstance(d["recommendations"], list)


class TestCartridgeImmutability:
    """Verificar que los cartuchos son inmutables (frozen dataclass)."""

    def test_swelling_cartridge_frozen(self) -> None:
        cart = SwellingPlasmonCartridge(
            liquid_limit=60.0,
            plasticity_index=30.0,
            parasitic_capacitance=0.5,
        )
        with pytest.raises(AttributeError):
            cart.liquid_limit = 100.0  # type: ignore[misc]

    def test_yielding_cartridge_frozen(self) -> None:
        cart = YieldingPhononCartridge(
            compression_index=0.3,
            void_ratio=1.0,
            viscous_drag=0.5,
        )
        with pytest.raises(AttributeError):
            cart.void_ratio = 2.0  # type: ignore[misc]

    def test_liquefaction_cartridge_frozen(self) -> None:
        cart = LiquefactionSolitonCartridge(
            shear_wave_velocity=100.0,
            susceptibility_index=0.5,
        )
        with pytest.raises(AttributeError):
            cart.shear_wave_velocity = 200.0  # type: ignore[misc]


class TestDiagnosticRuleEnum:
    """Verificar propiedades del enum DiagnosticRule."""

    def test_all_values_are_strings(self) -> None:
        for rule in DiagnosticRule:
            assert isinstance(rule.value, str)

    def test_uniqueness(self) -> None:
        values = [r.value for r in DiagnosticRule]
        assert len(values) == len(set(values))

    def test_enum_is_string_subclass(self) -> None:
        """DiagnosticRule hereda de str, permitiendo comparación directa."""
        assert isinstance(DiagnosticRule.SWELLING_POTENTIAL_DETECTED, str)

    def test_known_members(self) -> None:
        expected = {
            "swelling_potential_detected",
            "yielding_susceptibility_detected",
            "liquefaction_susceptibility_detected",
            "low_dynamic_rigidity_regime",
        }
        actual = {r.value for r in DiagnosticRule}
        assert actual == expected


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              §10  PARSEO DE PAYLOAD                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class TestPayloadParsing:
    """Pruebas del parseo de payload externo a SoilTensor."""

    def test_string_boolean_parsing(
        self, manifold: LithologicalManifold
    ) -> None:
        """El campo is_saturated como string debe parsearse correctamente."""
        payload = _make_payload(is_saturated="true")
        state = manifold(payload, {})
        # No debe lanzar; si is_saturated se parsea bien, el flujo continúa
        assert state is not None

    def test_integer_vs_parsing(
        self, manifold: LithologicalManifold
    ) -> None:
        """Vs como int debe convertirse a float sin error."""
        payload = _make_payload(vs=300)
        state = manifold(payload, {})
        assert state.payload["dynamic_rigidity_modulus_pa"] > 0

    def test_missing_optional_fields_use_defaults(
        self, manifold: LithologicalManifold
    ) -> None:
        """Payload con solo USCS debe usar defaults para todo lo demás."""
        payload: Dict[str, Any] = {"uscs": "GW"}
        state = manifold(payload, {})
        # Defaults: vs=300, e₀=0.5, etc.
        assert state.payload["dynamic_rigidity_modulus_pa"] > 0

    def test_none_boolean_raises(
        self, manifold: LithologicalManifold
    ) -> None:
        """None en is_saturated (si se pasa explícitamente) → error."""
        payload = _make_payload(is_saturated=None)
        with pytest.raises(LithologicalInputError):
            manifold(payload, {})

    def test_invalid_uscs_in_payload_raises(
        self, manifold: LithologicalManifold
    ) -> None:
        """USCS no reconocido en payload debe fallar en validación."""
        payload = _make_payload(uscs="XX")
        with pytest.raises(LithologicalInputError, match="no reconocida"):
            manifold(payload, {})


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              §11  CARTRIDGE POST-INIT VALIDATION                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class TestCartridgePostInitValidation:
    """
    Verificar que los cartuchos rechazan índices fuera de [0, 1]
    en su construcción (__post_init__).
    """

    def test_swelling_cartridge_rejects_index_above_one(self) -> None:
        with pytest.raises(LithologicalNumericalError):
            SwellingPlasmonCartridge(
                liquid_limit=60.0,
                plasticity_index=30.0,
                parasitic_capacitance=1.5,
            )

    def test_swelling_cartridge_rejects_negative_index(self) -> None:
        with pytest.raises(LithologicalNumericalError):
            SwellingPlasmonCartridge(
                liquid_limit=60.0,
                plasticity_index=30.0,
                parasitic_capacitance=-0.1,
            )

    def test_yielding_cartridge_rejects_invalid_drag(self) -> None:
        with pytest.raises(LithologicalNumericalError):
            YieldingPhononCartridge(
                compression_index=0.3,
                void_ratio=1.0,
                viscous_drag=2.0,
            )

    def test_liquefaction_cartridge_rejects_invalid_index(self) -> None:
        with pytest.raises(LithologicalNumericalError):
            LiquefactionSolitonCartridge(
                shear_wave_velocity=100.0,
                susceptibility_index=-0.5,
            )

    def test_valid_cartridges_construct_without_error(self) -> None:
        """Valores válidos no deben lanzar excepciones."""
        SwellingPlasmonCartridge(
            liquid_limit=60.0,
            plasticity_index=30.0,
            parasitic_capacitance=0.5,
        )
        YieldingPhononCartridge(
            compression_index=0.3,
            void_ratio=1.0,
            viscous_drag=0.0,
        )
        LiquefactionSolitonCartridge(
            shear_wave_velocity=100.0,
            susceptibility_index=1.0,
        )