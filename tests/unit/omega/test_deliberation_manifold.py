"""
Suite de pruebas exhaustiva para deliberation_manifold.py

Estrategia de testing:
----------------------
1. **Unit Tests puros**: Cada motor matemático se prueba en aislamiento,
   verificando propiedades analíticas (monotonía, acotamiento, continuidad).
2. **Tests de invariantes**: Propiedades que deben sostenerse para TODO input
   válido (finitud, rangos, normalización).
3. **Tests de frontera**: Valores extremos de ψ, ROI, anomalías, territorio.
4. **Tests de calibración**: Verifican que los escenarios canónicos producen
   los veredictos esperados según la documentación del manifold.
5. **Tests de integración**: Pipeline completo desde payload hasta OmegaResult.
6. **Tests de regresión**: Casos que revelaron bugs en versiones anteriores.

Convenciones:
- Fixtures reutilizables para configuraciones canónicas.
- Nombres: test_<unidad>_<condición>_<expectativa>
- Cada test verifica UNA propiedad (SRP de testing).
- pytest.approx para comparaciones de floats.
"""

from __future__ import annotations

import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_ROOT_PATH = Path(__file__).resolve().parent.parent
if str(_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(_ROOT_PATH))

from app.deliberation_manifold import (
    OmegaDeliberationManifold,
    OmegaDiagnostics,
    OmegaInputs,
    OmegaMetrics,
    OmegaResult,
    SynapticRegistry,
    ToonCartridge,
    _ANOMALY_COEFF_CYCLE,
    _ANOMALY_COEFF_ISOLATED,
    _ANOMALY_COEFF_STRESSED,
    _clamp,
    _DEFAULT_MAX_CARTRIDGES,
    _DEFAULT_MAX_CHARS,
    _EPSILON,
    _extract_profitability_index,
    _extract_topological_stability,
    _FRAGILITY_PENALTY_MAX_DELTA,
    _FRICTION_CLAMP_HIGH,
    _FRICTION_CLAMP_LOW,
    _FRICTION_FAVORABLE_THRESHOLD,
    _FRICTION_MODERATE_THRESHOLD,
    _FRICTION_WEIGHT_CLIMATE,
    _FRICTION_WEIGHT_LOGISTICS,
    _FRICTION_WEIGHT_SOCIAL,
    _GRAVITY_INFLECTION,
    _IMPROBABILITY_CLAMP_HIGH,
    _IMPROBABILITY_CLAMP_LOW,
    _IMPROBABILITY_SCALE_FACTOR,
    _interpret_friction,
    _interpret_psi,
    _interpret_roi,
    _interpret_stress,
    _PSI_CLAMP_HIGH,
    _PSI_CLAMP_LOW,
    _PSI_FRAGILE_THRESHOLD,
    _PSI_ROBUST_THRESHOLD,
    _ROI_CLAMP_HIGH,
    _ROI_CLAMP_LOW,
    _ROI_MODERATE_THRESHOLD,
    _ROI_WEAK_THRESHOLD,
    _safe_dict,
    _safe_float,
    _safe_int,
    _STRESS_HIGH_THRESHOLD,
    _STRESS_LOW_THRESHOLD,
    _STRESS_MODERATE_THRESHOLD,
    _VERDICT_THRESHOLD_CONDICIONAL,
    _VERDICT_THRESHOLD_PRECAUCION,
    _VERDICT_THRESHOLD_VIABLE,
)
from app.telemetry_schemas import VerdictLevel


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def manifold() -> OmegaDeliberationManifold:
    """Instancia limpia del manifold."""
    return OmegaDeliberationManifold()


@pytest.fixture
def neutral_inputs() -> OmegaInputs:
    """Inputs neutrales: sistema saludable sin anomalías ni territorio."""
    return OmegaInputs(
        psi=1.0,
        n_nodes=10,
        n_edges=15,
        cycle_count=0,
        isolated_count=0,
        stressed_count=0,
        roi=1.0,
        logistics_friction=1.0,
        social_friction=1.0,
        climate_entropy=1.0,
        territory_present=False,
    )


@pytest.fixture
def fragile_inputs() -> OmegaInputs:
    """Inputs con fragilidad estructural severa."""
    return OmegaInputs(
        psi=0.2,
        n_nodes=50,
        n_edges=80,
        cycle_count=5,
        isolated_count=10,
        stressed_count=3,
        roi=0.5,
        logistics_friction=1.0,
        social_friction=1.0,
        climate_entropy=1.0,
        territory_present=False,
    )


@pytest.fixture
def hostile_territory_inputs() -> OmegaInputs:
    """Inputs con territorio hostil activo."""
    return OmegaInputs(
        psi=0.8,
        n_nodes=30,
        n_edges=40,
        cycle_count=2,
        isolated_count=3,
        stressed_count=1,
        roi=1.2,
        logistics_friction=4.0,
        social_friction=3.5,
        climate_entropy=3.0,
        territory_present=True,
    )


@pytest.fixture
def ideal_inputs() -> OmegaInputs:
    """Inputs de sistema ideal: máxima estabilidad, buen ROI, sin problemas."""
    return OmegaInputs(
        psi=3.0,
        n_nodes=5,
        n_edges=4,
        cycle_count=0,
        isolated_count=0,
        stressed_count=0,
        roi=2.0,
        logistics_friction=1.0,
        social_friction=1.0,
        climate_entropy=1.0,
        territory_present=False,
    )


@pytest.fixture
def worst_case_inputs() -> OmegaInputs:
    """Inputs del peor caso: todo al máximo de riesgo."""
    return OmegaInputs(
        psi=_PSI_CLAMP_LOW,
        n_nodes=1000,
        n_edges=5000,
        cycle_count=50,
        isolated_count=100,
        stressed_count=30,
        roi=0.0,
        logistics_friction=_FRICTION_CLAMP_HIGH,
        social_friction=_FRICTION_CLAMP_HIGH,
        climate_entropy=_FRICTION_CLAMP_HIGH,
        territory_present=True,
    )


@pytest.fixture
def neutral_payload() -> Dict[str, Any]:
    """Payload neutro para construcción desde from_payload."""
    return {
        "tactics_state": {
            "pyramid_stability": 1.0,
            "n_nodes": 10,
            "n_edges": 15,
            "cycle_count": 0,
            "isolated_count": 0,
            "stressed_count": 0,
        },
        "strategy_state": {
            "profitability_index": 1.0,
        },
        "territory_state": {},
    }


@pytest.fixture
def full_payload() -> Dict[str, Any]:
    """Payload completo con territorio."""
    return {
        "tactics_state": {
            "pyramid_stability": 0.6,
            "n_nodes": 40,
            "n_edges": 60,
            "cycle_count": 3,
            "isolated_count": 5,
            "stressed_count": 2,
        },
        "strategy_state": {
            "profitability_index": 1.3,
        },
        "territory_state": {
            "logistics_friction": 2.5,
            "social_friction": 2.0,
            "climate_entropy": 1.5,
        },
    }


@pytest.fixture
def sample_cartridges() -> List[ToonCartridge]:
    """Conjunto de cartuchos para tests del SynapticRegistry."""
    return [
        ToonCartridge(name="alpha", domain="engineering", toon_payload="Alpha payload data", weight=3.0),
        ToonCartridge(name="beta", domain="finance", toon_payload="Beta payload data", weight=1.0),
        ToonCartridge(name="gamma", domain="logistics", toon_payload="Gamma payload data", weight=2.0),
        ToonCartridge(name="delta", domain="empty", toon_payload="", weight=5.0),
        ToonCartridge(name="epsilon", domain="whitespace", toon_payload="   ", weight=4.0),
    ]


# ============================================================================
# TEST: _clamp
# ============================================================================


class TestClamp:
    """Tests para acotamiento numérico."""

    def test_within_range(self):
        assert _clamp(0.5, 0.0, 1.0) == 0.5

    def test_at_lower_bound(self):
        assert _clamp(0.0, 0.0, 1.0) == 0.0

    def test_at_upper_bound(self):
        assert _clamp(1.0, 0.0, 1.0) == 1.0

    def test_below_lower(self):
        assert _clamp(-5.0, 0.0, 1.0) == 0.0

    def test_above_upper(self):
        assert _clamp(10.0, 0.0, 1.0) == 1.0

    def test_negative_range(self):
        assert _clamp(-3.0, -5.0, -1.0) == -3.0

    def test_zero_width_range(self):
        assert _clamp(99.0, 3.0, 3.0) == 3.0


# ============================================================================
# TEST: _safe_float
# ============================================================================


class TestSafeFloat:
    """Tests para conversión segura a float."""

    def test_valid_int(self):
        assert _safe_float(42, 0.0) == 42.0

    def test_valid_float(self):
        assert _safe_float(3.14, 0.0) == 3.14

    def test_valid_string(self):
        assert _safe_float("2.5", 0.0) == 2.5

    def test_none(self):
        assert _safe_float(None, -1.0) == -1.0

    def test_bool_true(self):
        assert _safe_float(True, 99.0) == 99.0

    def test_bool_false(self):
        assert _safe_float(False, 99.0) == 99.0

    def test_nan(self):
        assert _safe_float(float("nan"), 0.0) == 0.0

    def test_inf(self):
        assert _safe_float(float("inf"), 0.0) == 0.0

    def test_neg_inf(self):
        assert _safe_float(float("-inf"), 0.0) == 0.0

    def test_non_numeric_string(self):
        assert _safe_float("abc", 5.0) == 5.0

    def test_empty_string(self):
        assert _safe_float("", 5.0) == 5.0

    def test_list(self):
        assert _safe_float([1, 2], 0.0) == 0.0

    def test_zero(self):
        assert _safe_float(0, 99.0) == 0.0

    def test_negative(self):
        assert _safe_float(-42.5, 0.0) == -42.5


# ============================================================================
# TEST: _safe_int
# ============================================================================


class TestSafeInt:
    """Tests para conversión segura a int."""

    def test_valid_int(self):
        assert _safe_int(7, 0) == 7

    def test_float_truncated(self):
        assert _safe_int(3.9, 0) == 3

    def test_string_numeric(self):
        assert _safe_int("12", 0) == 12

    def test_none(self):
        assert _safe_int(None, -1) == -1

    def test_bool(self):
        assert _safe_int(True, 0) == 0

    def test_nan(self):
        assert _safe_int(float("nan"), 5) == 5

    def test_inf(self):
        assert _safe_int(float("inf"), 5) == 5

    def test_non_numeric(self):
        assert _safe_int("xyz", 99) == 99

    def test_negative(self):
        assert _safe_int(-10, 0) == -10


# ============================================================================
# TEST: _safe_dict
# ============================================================================


class TestSafeDict:
    """Tests para conversión segura a dict."""

    def test_valid_dict(self):
        d = {"key": "value"}
        assert _safe_dict(d) is d

    def test_none(self):
        assert _safe_dict(None) == {}

    def test_string(self):
        assert _safe_dict("not a dict") == {}

    def test_list(self):
        assert _safe_dict([1, 2, 3]) == {}

    def test_int(self):
        assert _safe_dict(42) == {}

    def test_empty_dict(self):
        assert _safe_dict({}) == {}


# ============================================================================
# TEST: _extract_topological_stability
# ============================================================================


class TestExtractTopologicalStability:
    """Tests para extracción de ψ."""

    def test_normal_value(self):
        result = _extract_topological_stability({"pyramid_stability": 1.0})
        assert result == 1.0

    def test_missing_key(self):
        result = _extract_topological_stability({})
        assert result == 1.0

    def test_below_clamp(self):
        result = _extract_topological_stability({"pyramid_stability": 0.001})
        assert result == _PSI_CLAMP_LOW

    def test_above_clamp(self):
        result = _extract_topological_stability({"pyramid_stability": 100.0})
        assert result == _PSI_CLAMP_HIGH

    def test_none_value(self):
        result = _extract_topological_stability({"pyramid_stability": None})
        assert result == 1.0

    def test_nan_value(self):
        result = _extract_topological_stability({"pyramid_stability": float("nan")})
        assert result == 1.0

    def test_string_numeric(self):
        result = _extract_topological_stability({"pyramid_stability": "2.5"})
        assert result == 2.5

    def test_negative_clamped(self):
        result = _extract_topological_stability({"pyramid_stability": -5.0})
        assert result == _PSI_CLAMP_LOW


# ============================================================================
# TEST: _extract_profitability_index
# ============================================================================


class TestExtractProfitabilityIndex:
    """Tests para extracción de ROI."""

    def test_normal_value(self):
        assert _extract_profitability_index({"profitability_index": 1.5}) == 1.5

    def test_missing_key(self):
        assert _extract_profitability_index({}) == 1.0

    def test_negative_clamped(self):
        assert _extract_profitability_index({"profitability_index": -1.0}) == _ROI_CLAMP_LOW

    def test_above_clamp(self):
        assert _extract_profitability_index({"profitability_index": 99.0}) == _ROI_CLAMP_HIGH

    def test_zero(self):
        assert _extract_profitability_index({"profitability_index": 0.0}) == 0.0

    def test_none(self):
        assert _extract_profitability_index({"profitability_index": None}) == 1.0


# ============================================================================
# TEST: Interpretación semántica
# ============================================================================


class TestInterpretPsi:
    """Tests para clasificación semántica de ψ."""

    def test_fragile(self):
        assert _interpret_psi(0.5) == "fragil"

    def test_fragile_boundary(self):
        assert _interpret_psi(_PSI_FRAGILE_THRESHOLD - 0.01) == "fragil"

    def test_stable(self):
        assert _interpret_psi(1.0) == "estable"

    def test_stable_boundary(self):
        assert _interpret_psi(_PSI_FRAGILE_THRESHOLD) == "estable"

    def test_robust(self):
        assert _interpret_psi(2.0) == "robusto"

    def test_robust_boundary(self):
        assert _interpret_psi(_PSI_ROBUST_THRESHOLD) == "robusto"


class TestInterpretRoi:
    """Tests para clasificación semántica de ROI."""

    def test_weak(self):
        assert _interpret_roi(0.5) == "retorno_debil"

    def test_moderate(self):
        assert _interpret_roi(1.2) == "retorno_moderado"

    def test_strong(self):
        assert _interpret_roi(2.0) == "retorno_fuerte"

    def test_boundary_weak_moderate(self):
        assert _interpret_roi(_ROI_WEAK_THRESHOLD - 0.01) == "retorno_debil"
        assert _interpret_roi(_ROI_WEAK_THRESHOLD) == "retorno_moderado"

    def test_boundary_moderate_strong(self):
        assert _interpret_roi(_ROI_MODERATE_THRESHOLD - 0.01) == "retorno_moderado"
        assert _interpret_roi(_ROI_MODERATE_THRESHOLD) == "retorno_fuerte"


class TestInterpretFriction:
    """Tests para clasificación semántica de fricción territorial."""

    def test_favorable(self):
        assert _interpret_friction(1.0) == "territorio_favorable"

    def test_moderate(self):
        assert _interpret_friction(1.5) == "territorio_moderado"

    def test_hostile(self):
        assert _interpret_friction(3.0) == "territorio_hostil"

    def test_boundary_favorable_moderate(self):
        assert _interpret_friction(_FRICTION_FAVORABLE_THRESHOLD - 0.01) == "territorio_favorable"
        assert _interpret_friction(_FRICTION_FAVORABLE_THRESHOLD) == "territorio_moderado"

    def test_boundary_moderate_hostile(self):
        assert _interpret_friction(_FRICTION_MODERATE_THRESHOLD - 0.01) == "territorio_moderado"
        assert _interpret_friction(_FRICTION_MODERATE_THRESHOLD) == "territorio_hostil"


class TestInterpretStress:
    """Tests para clasificación semántica de estrés."""

    def test_low(self):
        assert _interpret_stress(0.3) == "tension_baja"

    def test_moderate(self):
        assert _interpret_stress(1.0) == "tension_moderada"

    def test_high(self):
        assert _interpret_stress(2.5) == "tension_alta"

    def test_critical(self):
        assert _interpret_stress(5.0) == "tension_critica"

    def test_boundary_low_moderate(self):
        assert _interpret_stress(_STRESS_LOW_THRESHOLD - 0.01) == "tension_baja"
        assert _interpret_stress(_STRESS_LOW_THRESHOLD) == "tension_moderada"

    def test_boundary_moderate_high(self):
        assert _interpret_stress(_STRESS_MODERATE_THRESHOLD - 0.01) == "tension_moderada"
        assert _interpret_stress(_STRESS_MODERATE_THRESHOLD) == "tension_alta"

    def test_boundary_high_critical(self):
        assert _interpret_stress(_STRESS_HIGH_THRESHOLD - 0.01) == "tension_alta"
        assert _interpret_stress(_STRESS_HIGH_THRESHOLD) == "tension_critica"


# ============================================================================
# TEST: ToonCartridge
# ============================================================================


class TestToonCartridge:
    """Tests para el dataclass de cartucho sináptico."""

    def test_valid_creation(self):
        tc = ToonCartridge(name="test", domain="eng", toon_payload="data")
        assert tc.name == "test"
        assert tc.weight == 1.0

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="nombre no vacío"):
            ToonCartridge(name="", domain="eng", toon_payload="data")

    def test_whitespace_name_raises(self):
        with pytest.raises(ValueError, match="nombre no vacío"):
            ToonCartridge(name="   ", domain="eng", toon_payload="data")

    def test_custom_weight(self):
        tc = ToonCartridge(name="t", domain="d", toon_payload="p", weight=5.0)
        assert tc.weight == 5.0

    def test_frozen(self):
        tc = ToonCartridge(name="t", domain="d", toon_payload="p")
        with pytest.raises(AttributeError):
            tc.name = "other"

    def test_negative_weight_sanitized(self):
        tc = ToonCartridge(name="t", domain="d", toon_payload="p", weight=-3.0)
        assert tc.weight >= 0.0

    def test_nan_weight_sanitized(self):
        tc = ToonCartridge(name="t", domain="d", toon_payload="p", weight=float("nan"))
        assert math.isfinite(tc.weight)
        assert tc.weight >= 0.0


# ============================================================================
# TEST: SynapticRegistry
# ============================================================================


class TestSynapticRegistry:
    """Tests para el registro de capacidades emergentes."""

    def test_empty_registry(self):
        sr = SynapticRegistry()
        assert sr.cartridge_count == 0
        assert sr.get_active_context() == "CONTEXTO_COGNITIVO|VACIO"

    def test_load_cartridge(self):
        sr = SynapticRegistry()
        tc = ToonCartridge(name="test", domain="eng", toon_payload="payload")
        sr.load_cartridge(tc)
        assert sr.cartridge_count == 1

    def test_load_non_cartridge_raises(self):
        sr = SynapticRegistry()
        with pytest.raises(TypeError, match="ToonCartridge"):
            sr.load_cartridge("not a cartridge")

    def test_load_duplicate_replaces(self):
        sr = SynapticRegistry()
        tc1 = ToonCartridge(name="test", domain="v1", toon_payload="first")
        tc2 = ToonCartridge(name="test", domain="v2", toon_payload="second")
        sr.load_cartridge(tc1)
        sr.load_cartridge(tc2)
        assert sr.cartridge_count == 1
        context = sr.get_active_context()
        assert "second" in context

    def test_ordering_by_weight(self, sample_cartridges):
        sr = SynapticRegistry()
        for tc in sample_cartridges:
            sr.load_cartridge(tc)
        context = sr.get_active_context()
        lines = context.split("\n")
        # delta y epsilon tienen payloads vacíos/whitespace → omitidos
        # Orden por peso: alpha(3.0), gamma(2.0), beta(1.0)
        assert lines[0] == "Alpha payload data"
        assert lines[1] == "Gamma payload data"
        assert lines[2] == "Beta payload data"

    def test_empty_payload_skipped(self, sample_cartridges):
        sr = SynapticRegistry()
        for tc in sample_cartridges:
            sr.load_cartridge(tc)
        context = sr.get_active_context()
        # delta tiene payload vacío, epsilon tiene whitespace → ambos omitidos
        assert "delta" not in context.lower()
        assert context.strip()  # No vacío

    def test_max_items(self, sample_cartridges):
        sr = SynapticRegistry()
        for tc in sample_cartridges:
            sr.load_cartridge(tc)
        context = sr.get_active_context(max_items=2)
        lines = [l for l in context.split("\n") if l.strip()]
        assert len(lines) == 2

    def test_max_items_zero(self, sample_cartridges):
        sr = SynapticRegistry()
        for tc in sample_cartridges:
            sr.load_cartridge(tc)
        context = sr.get_active_context(max_items=0)
        assert context == "CONTEXTO_COGNITIVO|VACIO"

    def test_max_chars_respected(self):
        sr = SynapticRegistry()
        sr.load_cartridge(
            ToonCartridge(name="a", domain="d", toon_payload="12345", weight=2.0)
        )
        sr.load_cartridge(
            ToonCartridge(name="b", domain="d", toon_payload="67890", weight=1.0)
        )
        # "12345" = 5 chars, "\n" = 1, "67890" = 5 → total 11
        # Con max_chars=10, solo cabe el primero
        context = sr.get_active_context(max_chars=10)
        assert context == "12345"

    def test_max_chars_exact_fit(self):
        sr = SynapticRegistry()
        sr.load_cartridge(
            ToonCartridge(name="a", domain="d", toon_payload="12345", weight=2.0)
        )
        sr.load_cartridge(
            ToonCartridge(name="b", domain="d", toon_payload="67890", weight=1.0)
        )
        # Exactamente 11 caracteres: "12345\n67890"
        context = sr.get_active_context(max_chars=11)
        assert context == "12345\n67890"

    def test_max_chars_zero(self):
        sr = SynapticRegistry()
        sr.load_cartridge(
            ToonCartridge(name="a", domain="d", toon_payload="data", weight=1.0)
        )
        context = sr.get_active_context(max_chars=0)
        assert context == "CONTEXTO_COGNITIVO|VACIO"

    def test_max_chars_none_unlimited(self, sample_cartridges):
        sr = SynapticRegistry()
        for tc in sample_cartridges:
            sr.load_cartridge(tc)
        context = sr.get_active_context(max_chars=None)
        assert len(context) > 0
        assert context != "CONTEXTO_COGNITIVO|VACIO"


# ============================================================================
# TEST: OmegaInputs
# ============================================================================


class TestOmegaInputs:
    """Tests para coordenadas de entrada al manifold."""

    def test_default_values(self):
        oi = OmegaInputs()
        assert oi.psi == 1.0
        assert oi.n_nodes == 1
        assert oi.n_edges == 1
        assert oi.roi == 1.0
        assert oi.territory_present is False

    def test_frozen(self):
        oi = OmegaInputs()
        with pytest.raises(AttributeError):
            oi.psi = 2.0

    def test_from_payload_neutral(self, neutral_payload):
        oi = OmegaInputs.from_payload(neutral_payload)
        assert oi.psi == 1.0
        assert oi.n_nodes == 10
        assert oi.roi == 1.0
        assert oi.territory_present is False
        assert oi.logistics_friction == 1.0

    def test_from_payload_full(self, full_payload):
        oi = OmegaInputs.from_payload(full_payload)
        assert oi.psi == 0.6
        assert oi.n_nodes == 40
        assert oi.cycle_count == 3
        assert oi.roi == 1.3
        assert oi.territory_present is True
        assert oi.logistics_friction == 2.5

    def test_from_payload_empty(self):
        oi = OmegaInputs.from_payload({})
        assert oi.psi == 1.0
        assert oi.roi == 1.0
        assert oi.territory_present is False

    def test_from_payload_none(self):
        oi = OmegaInputs.from_payload(None)
        assert oi.psi == 1.0
        assert oi.territory_present is False

    def test_territory_empty_dict_is_absent(self):
        """Dict vacío de territorio → territory_present = False."""
        payload = {"territory_state": {}}
        oi = OmegaInputs.from_payload(payload)
        assert oi.territory_present is False
        assert oi.logistics_friction == 1.0
        assert oi.social_friction == 1.0
        assert oi.climate_entropy == 1.0

    def test_territory_present_with_data(self):
        payload = {"territory_state": {"logistics_friction": 3.0}}
        oi = OmegaInputs.from_payload(payload)
        assert oi.territory_present is True
        assert oi.logistics_friction == 3.0

    def test_psi_clamped_low(self):
        payload = {"tactics_state": {"pyramid_stability": -100.0}}
        oi = OmegaInputs.from_payload(payload)
        assert oi.psi == _PSI_CLAMP_LOW

    def test_psi_clamped_high(self):
        payload = {"tactics_state": {"pyramid_stability": 999.0}}
        oi = OmegaInputs.from_payload(payload)
        assert oi.psi == _PSI_CLAMP_HIGH

    def test_roi_clamped(self):
        payload = {"strategy_state": {"profitability_index": -5.0}}
        oi = OmegaInputs.from_payload(payload)
        assert oi.roi == _ROI_CLAMP_LOW

    def test_n_nodes_minimum_one(self):
        payload = {"tactics_state": {"n_nodes": 0}}
        oi = OmegaInputs.from_payload(payload)
        assert oi.n_nodes >= 1

    def test_n_edges_minimum_one(self):
        payload = {"tactics_state": {"n_edges": -5}}
        oi = OmegaInputs.from_payload(payload)
        assert oi.n_edges >= 1

    def test_counts_non_negative(self):
        payload = {
            "tactics_state": {
                "cycle_count": -3,
                "isolated_count": -1,
                "stressed_count": -2,
            }
        }
        oi = OmegaInputs.from_payload(payload)
        assert oi.cycle_count >= 0
        assert oi.isolated_count >= 0
        assert oi.stressed_count >= 0

    def test_friction_clamped(self):
        payload = {
            "territory_state": {
                "logistics_friction": 99.0,
                "social_friction": -1.0,
                "climate_entropy": 10.0,
            }
        }
        oi = OmegaInputs.from_payload(payload)
        assert oi.logistics_friction == _FRICTION_CLAMP_HIGH
        assert oi.social_friction == _FRICTION_CLAMP_LOW
        assert oi.climate_entropy == _FRICTION_CLAMP_HIGH

    def test_stores_raw_data(self, full_payload):
        oi = OmegaInputs.from_payload(full_payload)
        assert oi.topo_data == full_payload["tactics_state"]
        assert oi.fin_data == full_payload["strategy_state"]
        assert oi.territory_data == full_payload["territory_state"]

    def test_corrupted_payload_uses_defaults(self):
        payload = {
            "tactics_state": "not a dict",
            "strategy_state": None,
            "territory_state": [1, 2, 3],
        }
        oi = OmegaInputs.from_payload(payload)
        assert oi.psi == 1.0
        assert oi.roi == 1.0
        assert oi.territory_present is False


# ============================================================================
# TEST: MOTORES MATEMÁTICOS INTERNOS
# ============================================================================


class TestComputeFragilityNormalized:
    """Tests para fragilidad normalizada."""

    def test_output_range(self, manifold):
        """Para cualquier ψ válido, fragility_norm ∈ [0, 1]."""
        test_values = [_PSI_CLAMP_LOW, 0.1, 0.5, 1.0, 2.0, 3.0, _PSI_CLAMP_HIGH]
        for psi in test_values:
            result = manifold._compute_fragility_normalized(psi)
            assert 0.0 <= result <= 1.0, f"ψ={psi} → fragility={result}"

    def test_monotonically_decreasing(self, manifold):
        """Mayor ψ → menor fragilidad."""
        psi_values = [0.1, 0.5, 1.0, 2.0, 5.0]
        results = [manifold._compute_fragility_normalized(psi) for psi in psi_values]
        for i in range(len(results) - 1):
            assert results[i] > results[i + 1], (
                f"No monotónica: f({psi_values[i]})={results[i]} "
                f"≤ f({psi_values[i+1]})={results[i+1]}"
            )

    def test_maximum_at_psi_min(self, manifold):
        result = manifold._compute_fragility_normalized(_PSI_CLAMP_LOW)
        assert result == pytest.approx(1.0)

    def test_near_zero_at_psi_max(self, manifold):
        result = manifold._compute_fragility_normalized(_PSI_CLAMP_HIGH)
        assert result < 0.1

    def test_neutral_psi_low_fragility(self, manifold):
        result = manifold._compute_fragility_normalized(1.0)
        assert result < 0.15

    def test_epsilon_protection(self, manifold):
        """ψ = 0 no debe producir error."""
        result = manifold._compute_fragility_normalized(0.0)
        assert math.isfinite(result)
        assert 0.0 <= result <= 1.0

    def test_negative_psi_handled(self, manifold):
        result = manifold._compute_fragility_normalized(-5.0)
        assert math.isfinite(result)


class TestNormalizeRoi:
    """Tests para normalización de ROI."""

    def test_zero(self, manifold):
        assert manifold._normalize_roi(0.0) == 0.0

    def test_max(self, manifold):
        assert manifold._normalize_roi(_ROI_CLAMP_HIGH) == 1.0

    def test_neutral(self, manifold):
        result = manifold._normalize_roi(1.0)
        assert result == pytest.approx(1.0 / _ROI_CLAMP_HIGH)

    def test_above_max_clamped(self, manifold):
        assert manifold._normalize_roi(10.0) == 1.0

    def test_negative_clamped(self, manifold):
        assert manifold._normalize_roi(-1.0) == 0.0


class TestComputeMisalignment:
    """Tests para desalineación estructura-finanzas."""

    def test_zero_when_equal(self, manifold):
        assert manifold._compute_misalignment(0.5, 0.5) == 0.0

    def test_symmetric(self, manifold):
        assert manifold._compute_misalignment(0.3, 0.7) == manifold._compute_misalignment(0.7, 0.3)

    def test_range(self, manifold):
        result = manifold._compute_misalignment(0.0, 1.0)
        assert 0.0 <= result <= 1.0

    def test_maximum(self, manifold):
        assert manifold._compute_misalignment(0.0, 1.0) == 1.0


class TestComputeGravityCoupling:
    """Tests para coupling gravitacional."""

    def test_at_inflection(self, manifold):
        result = manifold._compute_gravity_coupling(_GRAVITY_INFLECTION)
        assert result == pytest.approx(1.0)

    def test_below_inflection(self, manifold):
        result = manifold._compute_gravity_coupling(0.0)
        assert result < 1.0
        assert result > 0.0

    def test_above_inflection(self, manifold):
        result = manifold._compute_gravity_coupling(1.0)
        assert result > 1.0

    def test_monotonically_increasing(self, manifold):
        values = [0.0, 0.25, 0.5, 0.75, 1.0]
        results = [manifold._compute_gravity_coupling(v) for v in values]
        for i in range(len(results) - 1):
            assert results[i] < results[i + 1]

    def test_bounded(self, manifold):
        """coupling = 1 + tanh(x) ∈ (0, 2)."""
        for val in [0.0, 0.5, 1.0]:
            result = manifold._compute_gravity_coupling(val)
            assert 0.0 < result < 2.0


class TestComputeInternalTension:
    """Tests para tensión interna."""

    def test_zero_misalignment(self, manifold):
        assert manifold._compute_internal_tension(0.0, 1.5) == 0.0

    def test_positive_result(self, manifold):
        result = manifold._compute_internal_tension(0.5, 1.2)
        assert result == pytest.approx(0.6)

    def test_never_negative(self, manifold):
        result = manifold._compute_internal_tension(0.0, 0.0)
        assert result >= 0.0


class TestComputeExternalFriction:
    """Tests para fricción territorial."""

    def test_no_territory(self, neutral_inputs):
        manifold = OmegaDeliberationManifold()
        result = manifold._compute_external_friction(neutral_inputs)
        assert result == 1.0

    def test_with_territory(self, hostile_territory_inputs):
        manifold = OmegaDeliberationManifold()
        result = manifold._compute_external_friction(hostile_territory_inputs)
        expected = (
            4.0 * _FRICTION_WEIGHT_LOGISTICS
            + 3.5 * _FRICTION_WEIGHT_SOCIAL
            + 3.0 * _FRICTION_WEIGHT_CLIMATE
        )
        assert result == pytest.approx(expected)

    def test_minimum_is_one(self):
        manifold = OmegaDeliberationManifold()
        inputs = OmegaInputs(
            logistics_friction=0.5,
            social_friction=0.3,
            climate_entropy=0.2,
            territory_present=True,
        )
        result = manifold._compute_external_friction(inputs)
        assert result >= 1.0

    def test_weights_sum_to_one(self):
        total = _FRICTION_WEIGHT_LOGISTICS + _FRICTION_WEIGHT_SOCIAL + _FRICTION_WEIGHT_CLIMATE
        assert total == pytest.approx(1.0)


class TestComputeAnomalyPressure:
    """Tests para presión anómala."""

    def test_no_anomalies(self, manifold, neutral_inputs):
        result = manifold._compute_anomaly_pressure(neutral_inputs)
        assert result == 1.0

    def test_with_anomalies(self, manifold, fragile_inputs):
        result = manifold._compute_anomaly_pressure(fragile_inputs)
        expected = 1.0 + (
            _ANOMALY_COEFF_CYCLE * 5
            + _ANOMALY_COEFF_ISOLATED * 10
            + _ANOMALY_COEFF_STRESSED * 3
        )
        assert result == pytest.approx(expected)

    def test_always_gte_one(self, manifold, neutral_inputs):
        result = manifold._compute_anomaly_pressure(neutral_inputs)
        assert result >= 1.0


class TestComputeCombinatorialScale:
    """Tests para escala combinatoria."""

    def test_minimum_graph(self, manifold):
        inputs = OmegaInputs(n_nodes=1, n_edges=1)
        result = manifold._compute_combinatorial_scale(inputs)
        assert result == pytest.approx(1.0)  # log10(10) = 1.0

    def test_larger_graph(self, manifold):
        inputs = OmegaInputs(n_nodes=100, n_edges=500)
        result = manifold._compute_combinatorial_scale(inputs)
        expected = math.log10(100 * 500)
        assert result == pytest.approx(expected)

    def test_always_gte_one(self, manifold):
        inputs = OmegaInputs(n_nodes=1, n_edges=1)
        assert manifold._compute_combinatorial_scale(inputs) >= 1.0


class TestComputeFrictionScale:
    """Tests para escala de fricción."""

    def test_neutral_friction(self, manifold):
        assert manifold._compute_friction_scale(1.0) == 1.0

    def test_high_friction(self, manifold):
        result = manifold._compute_friction_scale(4.0)
        assert result == pytest.approx(2.0)

    def test_below_one_clamped(self, manifold):
        result = manifold._compute_friction_scale(0.5)
        assert result == 1.0

    def test_sublinear(self, manifold):
        """sqrt crece más lentamente que la identidad."""
        for f in [2.0, 4.0, 9.0]:
            assert manifold._compute_friction_scale(f) < f


class TestComputeImprobabilityLever:
    """Tests para palanca de improbabilidad."""

    def test_minimum_clamp(self, manifold):
        result = manifold._compute_improbability_lever(1.0, 1.0, 1.0)
        expected = (1.0 * 1.0 * 1.0) / _IMPROBABILITY_SCALE_FACTOR
        clamped = _clamp(expected, _IMPROBABILITY_CLAMP_LOW, _IMPROBABILITY_CLAMP_HIGH)
        assert result == pytest.approx(clamped)

    def test_maximum_clamp(self, manifold):
        result = manifold._compute_improbability_lever(100.0, 100.0, 100.0)
        assert result == _IMPROBABILITY_CLAMP_HIGH

    def test_within_range(self, manifold):
        result = manifold._compute_improbability_lever(1.5, 2.0, 1.2)
        assert _IMPROBABILITY_CLAMP_LOW <= result <= _IMPROBABILITY_CLAMP_HIGH


class TestComputeFragilityPenalty:
    """Tests para penalización por fragilidad."""

    def test_healthy_psi(self, manifold):
        assert manifold._compute_fragility_penalty(1.0) == 1.0

    def test_high_psi(self, manifold):
        assert manifold._compute_fragility_penalty(3.0) == 1.0

    def test_low_psi(self, manifold):
        result = manifold._compute_fragility_penalty(0.5)
        expected = 1.0 + (0.5 * _FRAGILITY_PENALTY_MAX_DELTA)
        assert result == pytest.approx(expected)

    def test_zero_psi(self, manifold):
        result = manifold._compute_fragility_penalty(0.0)
        expected = 1.0 + _FRAGILITY_PENALTY_MAX_DELTA
        assert result == pytest.approx(expected)

    def test_maximum_penalty(self, manifold):
        result = manifold._compute_fragility_penalty(0.0)
        assert result == pytest.approx(1.0 + _FRAGILITY_PENALTY_MAX_DELTA)

    def test_negative_psi_capped(self, manifold):
        result = manifold._compute_fragility_penalty(-5.0)
        assert result == pytest.approx(1.0 + _FRAGILITY_PENALTY_MAX_DELTA)

    def test_range(self, manifold):
        """Para todo ψ, penalty ∈ [1.0, 1 + δ_max]."""
        for psi in [_PSI_CLAMP_LOW, 0.0, 0.5, 0.99, 1.0, 2.0, 5.0]:
            result = manifold._compute_fragility_penalty(psi)
            assert 1.0 <= result <= 1.0 + _FRAGILITY_PENALTY_MAX_DELTA


class TestProjectToLattice:
    """Tests para proyección al retículo de veredictos."""

    def test_viable(self, manifold):
        assert manifold._project_to_lattice(0.0) == VerdictLevel.VIABLE
        assert manifold._project_to_lattice(0.74) == VerdictLevel.VIABLE

    def test_condicional(self, manifold):
        assert manifold._project_to_lattice(0.75) == VerdictLevel.CONDICIONAL
        assert manifold._project_to_lattice(1.74) == VerdictLevel.CONDICIONAL

    def test_precaucion(self, manifold):
        assert manifold._project_to_lattice(1.75) == VerdictLevel.PRECAUCION
        assert manifold._project_to_lattice(2.99) == VerdictLevel.PRECAUCION

    def test_rechazar(self, manifold):
        assert manifold._project_to_lattice(3.0) == VerdictLevel.RECHAZAR
        assert manifold._project_to_lattice(100.0) == VerdictLevel.RECHAZAR

    def test_boundaries_exact(self, manifold):
        assert manifold._project_to_lattice(_VERDICT_THRESHOLD_VIABLE - 0.001) == VerdictLevel.VIABLE
        assert manifold._project_to_lattice(_VERDICT_THRESHOLD_VIABLE) == VerdictLevel.CONDICIONAL
        assert manifold._project_to_lattice(_VERDICT_THRESHOLD_CONDICIONAL) == VerdictLevel.PRECAUCION
        assert manifold._project_to_lattice(_VERDICT_THRESHOLD_PRECAUCION) == VerdictLevel.RECHAZAR

    def test_negative_stress(self, manifold):
        assert manifold._project_to_lattice(-1.0) == VerdictLevel.VIABLE


# ============================================================================
# TEST: _identify_dominant_risk_axis
# ============================================================================


class TestIdentifyDominantRiskAxis:
    """Tests para identificación del eje de riesgo dominante."""

    def test_balanced(self, manifold):
        metrics = OmegaMetrics(
            fragility_norm=0.1, roi_norm=0.2,
            misalignment=0.1, gravity_coupling=1.0,
            internal_tension=0.0, external_friction=1.0,
            anomaly_pressure=1.0, combinatorial_scale=1.0,
            friction_scale=1.0, improbability_lever=1.0,
            base_stress=0.0, fragility_penalty=1.0,
            total_stress=0.0, adjusted_stress=0.0,
        )
        axis, breakdown = manifold._identify_dominant_risk_axis(metrics)
        assert axis == "balanced"
        assert isinstance(breakdown, dict)

    def test_internal_dominant(self, manifold):
        metrics = OmegaMetrics(
            fragility_norm=0.8, roi_norm=0.2,
            misalignment=0.6, gravity_coupling=1.3,
            internal_tension=5.0, external_friction=1.0,
            anomaly_pressure=1.0, combinatorial_scale=1.0,
            friction_scale=1.0, improbability_lever=1.0,
            base_stress=5.0, fragility_penalty=1.0,
            total_stress=5.0, adjusted_stress=5.0,
        )
        axis, _ = manifold._identify_dominant_risk_axis(metrics)
        assert axis == "internal"

    def test_territory_dominant(self, manifold):
        metrics = OmegaMetrics(
            fragility_norm=0.1, roi_norm=0.2,
            misalignment=0.1, gravity_coupling=1.0,
            internal_tension=0.1, external_friction=4.0,
            anomaly_pressure=1.0, combinatorial_scale=1.0,
            friction_scale=1.0, improbability_lever=1.0,
            base_stress=0.4, fragility_penalty=1.0,
            total_stress=0.4, adjusted_stress=0.4,
        )
        axis, _ = manifold._identify_dominant_risk_axis(metrics)
        assert axis == "territory"

    def test_returns_breakdown(self, manifold):
        metrics = OmegaMetrics(
            fragility_norm=0.5, roi_norm=0.3,
            misalignment=0.2, gravity_coupling=1.1,
            internal_tension=0.22, external_friction=2.0,
            anomaly_pressure=1.2, combinatorial_scale=2.0,
            friction_scale=1.4, improbability_lever=2.5,
            base_stress=0.44, fragility_penalty=1.3,
            total_stress=1.1, adjusted_stress=1.43,
        )
        axis, breakdown = manifold._identify_dominant_risk_axis(metrics)
        assert "internal" in breakdown
        assert "territory" in breakdown
        assert "extremes" in breakdown
        assert "fragility" in breakdown

    def test_deterministic_on_tie(self, manifold):
        """Empates se resuelven lexicográficamente."""
        metrics = OmegaMetrics(
            fragility_norm=0.5, roi_norm=0.3,
            misalignment=0.2, gravity_coupling=1.0,
            internal_tension=1.0, external_friction=2.0,
            anomaly_pressure=1.0, combinatorial_scale=1.0,
            friction_scale=1.0, improbability_lever=2.0,
            base_stress=2.0, fragility_penalty=2.0,
            total_stress=4.0, adjusted_stress=8.0,
        )
        # territory = 2.0 - 1.0 = 1.0
        # extremes = 2.0 - 1.0 = 1.0
        # fragility = 2.0 - 1.0 = 1.0
        # internal = 1.0
        # Todos empatados en 1.0 → lexicográfico → "extremes"
        axis, _ = manifold._identify_dominant_risk_axis(metrics)
        assert axis == "extremes"

        # Verificar que llamadas sucesivas son idénticas
        axis2, _ = manifold._identify_dominant_risk_axis(metrics)
        assert axis2 == axis


# ============================================================================
# TEST: _compute_metrics (pipeline completo)
# ============================================================================


class TestComputeMetrics:
    """Tests para el pipeline de cálculo de métricas."""

    def test_neutral_inputs(self, manifold, neutral_inputs):
        metrics = manifold._compute_metrics(neutral_inputs)
        assert math.isfinite(metrics.adjusted_stress)
        assert metrics.external_friction == 1.0
        assert metrics.fragility_penalty == 1.0

    def test_all_metrics_finite(self, manifold, worst_case_inputs):
        """Ninguna métrica debe ser NaN o Inf, incluso en el peor caso."""
        metrics = manifold._compute_metrics(worst_case_inputs)
        for field_name in [
            "fragility_norm", "roi_norm", "misalignment", "gravity_coupling",
            "internal_tension", "external_friction", "anomaly_pressure",
            "combinatorial_scale", "friction_scale", "improbability_lever",
            "base_stress", "fragility_penalty", "total_stress", "adjusted_stress",
        ]:
            value = getattr(metrics, field_name)
            assert math.isfinite(value), f"{field_name} no es finito: {value}"

    def test_stress_nonnegative(self, manifold, worst_case_inputs):
        metrics = manifold._compute_metrics(worst_case_inputs)
        assert metrics.base_stress >= 0
        assert metrics.total_stress >= 0
        assert metrics.adjusted_stress >= 0

    def test_fragility_norm_in_range(self, manifold):
        for psi in [_PSI_CLAMP_LOW, 0.5, 1.0, 2.0, _PSI_CLAMP_HIGH]:
            inputs = OmegaInputs(psi=psi)
            metrics = manifold._compute_metrics(inputs)
            assert 0.0 <= metrics.fragility_norm <= 1.0

    def test_roi_norm_in_range(self, manifold):
        for roi in [_ROI_CLAMP_LOW, 0.5, 1.0, 3.0, _ROI_CLAMP_HIGH]:
            inputs = OmegaInputs(roi=roi)
            metrics = manifold._compute_metrics(inputs)
            assert 0.0 <= metrics.roi_norm <= 1.0

    def test_misalignment_in_range(self, manifold):
        inputs = OmegaInputs(psi=_PSI_CLAMP_LOW, roi=_ROI_CLAMP_HIGH)
        metrics = manifold._compute_metrics(inputs)
        assert 0.0 <= metrics.misalignment <= 1.0

    def test_improbability_lever_clamped(self, manifold, worst_case_inputs):
        metrics = manifold._compute_metrics(worst_case_inputs)
        assert _IMPROBABILITY_CLAMP_LOW <= metrics.improbability_lever <= _IMPROBABILITY_CLAMP_HIGH

    def test_fragility_penalty_range(self, manifold):
        for psi in [0.0, 0.5, 0.99, 1.0, 3.0]:
            inputs = OmegaInputs(psi=psi)
            metrics = manifold._compute_metrics(inputs)
            assert 1.0 <= metrics.fragility_penalty <= 1.0 + _FRAGILITY_PENALTY_MAX_DELTA

    def test_stress_formula_consistency(self, manifold, neutral_inputs):
        """Verifica que σ* = T_int × F_ext × Λ × P_frag."""
        metrics = manifold._compute_metrics(neutral_inputs)
        expected_base = metrics.internal_tension * metrics.external_friction
        expected_total = expected_base * metrics.improbability_lever
        expected_adjusted = expected_total * metrics.fragility_penalty
        assert metrics.base_stress == pytest.approx(expected_base)
        assert metrics.total_stress == pytest.approx(expected_total)
        assert metrics.adjusted_stress == pytest.approx(expected_adjusted)

    def test_stress_monotonic_with_anomalies(self, manifold):
        """Más anomalías → mayor estrés (ceteris paribus)."""
        base = OmegaInputs(psi=0.8, roi=1.2, n_nodes=30, n_edges=40)
        inputs_low = OmegaInputs(
            psi=0.8, roi=1.2, n_nodes=30, n_edges=40,
            cycle_count=0, isolated_count=0, stressed_count=0,
        )
        inputs_high = OmegaInputs(
            psi=0.8, roi=1.2, n_nodes=30, n_edges=40,
            cycle_count=10, isolated_count=20, stressed_count=5,
        )
        stress_low = manifold._compute_metrics(inputs_low).adjusted_stress
        stress_high = manifold._compute_metrics(inputs_high).adjusted_stress
        assert stress_high >= stress_low

    def test_stress_monotonic_with_fragility(self, manifold):
        """Menor ψ → mayor estrés (ceteris paribus)."""
        inputs_stable = OmegaInputs(psi=2.0, roi=1.0, n_nodes=10, n_edges=15)
        inputs_fragile = OmegaInputs(psi=0.3, roi=1.0, n_nodes=10, n_edges=15)
        stress_stable = manifold._compute_metrics(inputs_stable).adjusted_stress
        stress_fragile = manifold._compute_metrics(inputs_fragile).adjusted_stress
        assert stress_fragile >= stress_stable


# ============================================================================
# TEST: _collapse (orquestación)
# ============================================================================


class TestCollapse:
    """Tests para la orquestación del colapso."""

    def test_returns_omega_result(self, manifold, neutral_inputs):
        result = manifold._collapse(neutral_inputs)
        assert isinstance(result, OmegaResult)
        assert isinstance(result.metrics, OmegaMetrics)
        assert isinstance(result.diagnostics, OmegaDiagnostics)
        assert isinstance(result.verdict, VerdictLevel)

    def test_neutral_is_viable(self, manifold, neutral_inputs):
        result = manifold._collapse(neutral_inputs)
        assert result.verdict == VerdictLevel.VIABLE

    def test_ideal_is_viable(self, manifold, ideal_inputs):
        result = manifold._collapse(ideal_inputs)
        assert result.verdict == VerdictLevel.VIABLE

    def test_worst_case_is_rechazar(self, manifold, worst_case_inputs):
        result = manifold._collapse(worst_case_inputs)
        assert result.verdict == VerdictLevel.RECHAZAR

    def test_fragile_escalates(self, manifold, fragile_inputs):
        result = manifold._collapse(fragile_inputs)
        assert result.verdict in (
            VerdictLevel.CONDICIONAL,
            VerdictLevel.PRECAUCION,
            VerdictLevel.RECHAZAR,
        )

    def test_hostile_territory_increases_stress(self, manifold, hostile_territory_inputs):
        result = manifold._collapse(hostile_territory_inputs)
        # Con territorio hostil, no debería ser VIABLE
        assert result.verdict != VerdictLevel.VIABLE or result.metrics.adjusted_stress > 0

    def test_pure_function(self, manifold, neutral_inputs):
        """_collapse es función pura: mismos inputs → mismos outputs."""
        r1 = manifold._collapse(neutral_inputs)
        r2 = manifold._collapse(neutral_inputs)
        assert r1.verdict == r2.verdict
        assert r1.metrics.adjusted_stress == r2.metrics.adjusted_stress


# ============================================================================
# TEST: OmegaDiagnostics
# ============================================================================


class TestOmegaDiagnostics:
    """Tests para el contenedor de diagnósticos."""

    def test_to_dict(self, manifold, neutral_inputs):
        result = manifold._collapse(neutral_inputs)
        diag_dict = result.diagnostics.to_dict()
        assert isinstance(diag_dict, dict)
        assert "topology_status" in diag_dict
        assert "financial_status" in diag_dict
        assert "territory_status" in diag_dict
        assert "stress_status" in diag_dict
        assert "dominant_risk_axis" in diag_dict
        assert "summary" in diag_dict
        assert "inputs_snapshot" in diag_dict
        assert "derived_snapshot" in diag_dict

    def test_inputs_snapshot_complete(self, manifold, neutral_inputs):
        result = manifold._collapse(neutral_inputs)
        snap = result.diagnostics.inputs_snapshot
        expected_keys = {
            "psi", "roi", "n_nodes", "n_edges",
            "cycle_count", "isolated_count", "stressed_count",
            "territory_present", "logistics_friction",
            "social_friction", "climate_entropy",
        }
        assert expected_keys.issubset(set(snap.keys()))

    def test_derived_snapshot_complete(self, manifold, neutral_inputs):
        result = manifold._collapse(neutral_inputs)
        snap = result.diagnostics.derived_snapshot
        expected_keys = {
            "fragility_norm", "roi_norm", "misalignment", "gravity_coupling",
            "internal_tension", "external_friction", "anomaly_pressure",
            "combinatorial_scale", "friction_scale", "improbability_lever",
            "base_stress", "fragility_penalty", "total_stress", "adjusted_stress",
        }
        assert expected_keys.issubset(set(snap.keys()))

    def test_summary_contains_verdict(self, manifold, neutral_inputs):
        result = manifold._collapse(neutral_inputs)
        assert "Veredicto=" in result.diagnostics.summary

    def test_risk_contribution_breakdown_present(self, manifold, neutral_inputs):
        result = manifold._collapse(neutral_inputs)
        breakdown = result.diagnostics.risk_contribution_breakdown
        assert isinstance(breakdown, dict)
        assert "internal" in breakdown
        assert "territory" in breakdown
        assert "extremes" in breakdown
        assert "fragility" in breakdown

    def test_snapshots_values_finite(self, manifold, worst_case_inputs):
        result = manifold._collapse(worst_case_inputs)
        for key, val in result.diagnostics.derived_snapshot.items():
            if isinstance(val, float):
                assert math.isfinite(val), f"derived_snapshot[{key}] = {val}"
        for key, val in result.diagnostics.inputs_snapshot.items():
            if isinstance(val, float):
                assert math.isfinite(val), f"inputs_snapshot[{key}] = {val}"


# ============================================================================
# TEST: OmegaResult.to_payload
# ============================================================================


class TestOmegaResultToPayload:
    """Tests para serialización del resultado."""

    def test_structure(self, manifold, neutral_inputs):
        result = manifold._collapse(neutral_inputs)
        payload = result.to_payload(synaptic_context_toon="test_context")
        assert "omega_metrics" in payload
        assert "verdict" in payload
        assert "synaptic_context_toon" in payload
        assert "omega_diagnostics" in payload
        assert payload["synaptic_context_toon"] == "test_context"

    def test_metrics_keys(self, manifold, neutral_inputs):
        result = manifold._collapse(neutral_inputs)
        payload = result.to_payload(synaptic_context_toon="ctx")
        metrics = payload["omega_metrics"]
        expected_keys = {
            "topological_stability", "fragility_norm", "roi_norm",
            "internal_tension", "external_friction", "improbability_lever",
            "base_stress", "total_stress", "adjusted_stress",
            "misalignment", "gravity_coupling", "fragility_penalty",
            "anomaly_pressure", "combinatorial_scale", "friction_scale",
        }
        assert expected_keys == set(metrics.keys())

    def test_all_metrics_rounded(self, manifold, neutral_inputs):
        result = manifold._collapse(neutral_inputs)
        payload = result.to_payload(synaptic_context_toon="ctx")
        for key, val in payload["omega_metrics"].items():
            assert isinstance(val, float), f"{key} no es float"
            # Verificar que tiene a lo sumo 6 decimales
            as_str = f"{val:.6f}"
            assert float(as_str) == val, f"{key} tiene más de 6 decimales"

    def test_all_metrics_finite(self, manifold, worst_case_inputs):
        result = manifold._collapse(worst_case_inputs)
        payload = result.to_payload(synaptic_context_toon="ctx")
        for key, val in payload["omega_metrics"].items():
            assert math.isfinite(val), f"omega_metrics[{key}] = {val}"


# ============================================================================
# TEST: OmegaDeliberationManifold.__call__ (integración)
# ============================================================================


class TestManifoldCall:
    """Tests de integración para el morfismo completo."""

    def _make_state(
        self, payload: Dict[str, Any], is_success: bool = True
    ) -> MagicMock:
        """Crea un mock de CategoricalState."""
        state = MagicMock()
        state.is_success = is_success
        state.payload = payload
        state.with_update = MagicMock(return_value=MagicMock(spec=["is_success"]))
        state.with_error = MagicMock(return_value=MagicMock(spec=["is_success"]))
        return state

    def test_failed_state_passthrough(self, manifold):
        state = self._make_state({}, is_success=False)
        result = manifold(state)
        assert result is state
        state.with_update.assert_not_called()

    def test_successful_collapse(self, manifold, neutral_payload):
        state = self._make_state(neutral_payload)
        manifold(state)
        state.with_update.assert_called_once()
        call_kwargs = state.with_update.call_args
        new_payload = call_kwargs.kwargs.get("new_payload") or call_kwargs[1].get("new_payload")
        assert "omega_state" in new_payload

    def test_collapse_with_full_payload(self, manifold, full_payload):
        state = self._make_state(full_payload)
        manifold(state)
        state.with_update.assert_called_once()

    def test_empty_payload_succeeds(self, manifold):
        state = self._make_state({})
        manifold(state)
        state.with_update.assert_called_once()

    def test_none_payload_succeeds(self, manifold):
        state = self._make_state(None)
        state.payload = None
        manifold(state)
        state.with_update.assert_called_once()

    def test_non_dict_payload_succeeds(self, manifold):
        state = self._make_state("not a dict")
        state.payload = "not a dict"
        manifold(state)
        state.with_update.assert_called_once()

    def test_codomain_is_omega(self, manifold):
        from app.schemas import Stratum
        assert manifold.codomain == Stratum.OMEGA

    def test_domain_contains_tactics_and_strategy(self, manifold):
        from app.schemas import Stratum
        assert Stratum.TACTICS in manifold.domain
        assert Stratum.STRATEGY in manifold.domain

    def test_synaptic_context_injected(self, manifold, neutral_payload):
        tc = ToonCartridge(name="test", domain="d", toon_payload="injected_data")
        manifold.synaptic_registry.load_cartridge(tc)

        state = self._make_state(neutral_payload)
        manifold(state)

        call_kwargs = state.with_update.call_args
        new_payload = call_kwargs.kwargs.get("new_payload") or call_kwargs[1].get("new_payload")
        omega_state = new_payload["omega_state"]
        assert omega_state["synaptic_context_toon"] == "injected_data"

    def test_exception_returns_error_state(self, manifold):
        state = self._make_state({})
        # Forzar un error inyectando un from_payload que falle
        with patch.object(OmegaInputs, "from_payload", side_effect=RuntimeError("Boom")):
            manifold(state)
        state.with_error.assert_called_once()
        error_msg = state.with_error.call_args[0][0]
        assert "Boom" in error_msg

    def test_keyboard_interrupt_propagates(self, manifold):
        state = self._make_state({})
        with patch.object(OmegaInputs, "from_payload", side_effect=KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                manifold(state)

    def test_system_exit_propagates(self, manifold):
        state = self._make_state({})
        with patch.object(OmegaInputs, "from_payload", side_effect=SystemExit):
            with pytest.raises(SystemExit):
                manifold(state)


# ============================================================================
# TEST: CALIBRACIÓN DE ESCENARIOS CANÓNICOS
# ============================================================================


class TestCalibration:
    """Tests de calibración que verifican el comportamiento esperado
    del manifold en escenarios representativos del dominio.

    Estos tests documentan el contrato semántico del sistema:
    qué tipo de proyectos deben recibir qué tipo de veredicto.
    """

    def test_perfect_project_is_viable(self, manifold):
        """Proyecto perfecto: estructura sólida, buen ROI, sin problemas."""
        inputs = OmegaInputs(
            psi=2.0, roi=1.8, n_nodes=10, n_edges=12,
            cycle_count=0, isolated_count=0, stressed_count=0,
            territory_present=False,
        )
        result = manifold._collapse(inputs)
        assert result.verdict == VerdictLevel.VIABLE

    def test_moderate_issues_is_condicional(self, manifold):
        """Proyecto con problemas moderados: algo de fragilidad y anomalías."""
        inputs = OmegaInputs(
            psi=0.7, roi=1.0, n_nodes=30, n_edges=40,
            cycle_count=2, isolated_count=3, stressed_count=1,
            territory_present=False,
        )
        result = manifold._collapse(inputs)
        assert result.verdict in (VerdictLevel.CONDICIONAL, VerdictLevel.PRECAUCION)

    def test_catastrophic_project_rejected(self, manifold):
        """Proyecto catastrófico: todo mal."""
        inputs = OmegaInputs(
            psi=_PSI_CLAMP_LOW, roi=0.0, n_nodes=500, n_edges=2000,
            cycle_count=30, isolated_count=50, stressed_count=20,
            logistics_friction=_FRICTION_CLAMP_HIGH,
            social_friction=_FRICTION_CLAMP_HIGH,
            climate_entropy=_FRICTION_CLAMP_HIGH,
            territory_present=True,
        )
        result = manifold._collapse(inputs)
        assert result.verdict == VerdictLevel.RECHAZAR

    def test_good_structure_bad_territory(self, manifold):
        """Estructura buena pero territorio muy hostil."""
        inputs = OmegaInputs(
            psi=1.5, roi=1.5, n_nodes=10, n_edges=12,
            cycle_count=0, isolated_count=0, stressed_count=0,
            logistics_friction=4.5, social_friction=4.0, climate_entropy=4.0,
            territory_present=True,
        )
        result = manifold._collapse(inputs)
        # Con buena estructura, el territorio no debería empujar a RECHAZAR
        # pero podría ser CONDICIONAL
        assert result.verdict in (VerdictLevel.VIABLE, VerdictLevel.CONDICIONAL)

    def test_bad_structure_good_territory(self, manifold):
        """Estructura débil pero territorio favorable."""
        inputs = OmegaInputs(
            psi=0.3, roi=0.8, n_nodes=40, n_edges=60,
            cycle_count=5, isolated_count=8, stressed_count=3,
            logistics_friction=1.0, social_friction=1.0, climate_entropy=1.0,
            territory_present=True,
        )
        result = manifold._collapse(inputs)
        # Sin ayuda del territorio, la estructura débil debería escalar
        assert result.verdict != VerdictLevel.VIABLE

    def test_territory_absence_is_neutral(self, manifold):
        """Sin territorio, fricción no amplifica el estrés."""
        base_inputs = OmegaInputs(
            psi=0.8, roi=1.2, n_nodes=20, n_edges=25,
            cycle_count=1, isolated_count=2, stressed_count=0,
            territory_present=False,
        )
        with_territory = OmegaInputs(
            psi=0.8, roi=1.2, n_nodes=20, n_edges=25,
            cycle_count=1, isolated_count=2, stressed_count=0,
            logistics_friction=3.0, social_friction=3.0, climate_entropy=3.0,
            territory_present=True,
        )
        stress_without = manifold._compute_metrics(base_inputs).adjusted_stress
        stress_with = manifold._compute_metrics(with_territory).adjusted_stress
        assert stress_with >= stress_without

    def test_more_anomalies_never_improve_verdict(self, manifold):
        """Agregar anomalías nunca mejora el veredicto."""
        clean = OmegaInputs(
            psi=1.0, roi=1.0, n_nodes=30, n_edges=40,
            cycle_count=0, isolated_count=0, stressed_count=0,
        )
        dirty = OmegaInputs(
            psi=1.0, roi=1.0, n_nodes=30, n_edges=40,
            cycle_count=10, isolated_count=20, stressed_count=5,
        )
        clean_stress = manifold._compute_metrics(clean).adjusted_stress
        dirty_stress = manifold._compute_metrics(dirty).adjusted_stress
        assert dirty_stress >= clean_stress


# ============================================================================
# TEST: CONSTANTES MATEMÁTICAS
# ============================================================================


class TestMathematicalConstants:
    """Verifica invariantes de las constantes del módulo."""

    def test_verdict_thresholds_ordered(self):
        assert _VERDICT_THRESHOLD_VIABLE < _VERDICT_THRESHOLD_CONDICIONAL
        assert _VERDICT_THRESHOLD_CONDICIONAL < _VERDICT_THRESHOLD_PRECAUCION

    def test_verdict_thresholds_positive(self):
        assert _VERDICT_THRESHOLD_VIABLE > 0
        assert _VERDICT_THRESHOLD_CONDICIONAL > 0
        assert _VERDICT_THRESHOLD_PRECAUCION > 0

    def test_friction_weights_sum_to_one(self):
        total = _FRICTION_WEIGHT_LOGISTICS + _FRICTION_WEIGHT_SOCIAL + _FRICTION_WEIGHT_CLIMATE
        assert total == pytest.approx(1.0)

    def test_friction_weights_positive(self):
        assert _FRICTION_WEIGHT_LOGISTICS > 0
        assert _FRICTION_WEIGHT_SOCIAL > 0
        assert _FRICTION_WEIGHT_CLIMATE > 0

    def test_anomaly_coefficients_positive(self):
        assert _ANOMALY_COEFF_CYCLE > 0
        assert _ANOMALY_COEFF_ISOLATED > 0
        assert _ANOMALY_COEFF_STRESSED > 0

    def test_psi_clamp_range_valid(self):
        assert _PSI_CLAMP_LOW > 0
        assert _PSI_CLAMP_HIGH > _PSI_CLAMP_LOW

    def test_roi_clamp_range_valid(self):
        assert _ROI_CLAMP_LOW >= 0
        assert _ROI_CLAMP_HIGH > _ROI_CLAMP_LOW

    def test_improbability_clamp_valid(self):
        assert _IMPROBABILITY_CLAMP_LOW >= 1.0
        assert _IMPROBABILITY_CLAMP_HIGH > _IMPROBABILITY_CLAMP_LOW

    def test_fragility_penalty_max_positive(self):
        assert _FRAGILITY_PENALTY_MAX_DELTA > 0

    def test_epsilon_small_positive(self):
        assert _EPSILON > 0
        assert _EPSILON < 0.01

    def test_gravity_inflection_in_unit_interval(self):
        assert 0.0 < _GRAVITY_INFLECTION < 1.0

    def test_scale_factor_positive(self):
        assert _IMPROBABILITY_SCALE_FACTOR > 0

    def test_interpretation_thresholds_consistent_with_verdict(self):
        """Los umbrales de interpretación de estrés deben coincidir con veredictos."""
        assert _STRESS_LOW_THRESHOLD == _VERDICT_THRESHOLD_VIABLE
        assert _STRESS_MODERATE_THRESHOLD == _VERDICT_THRESHOLD_CONDICIONAL
        assert _STRESS_HIGH_THRESHOLD == _VERDICT_THRESHOLD_PRECAUCION

    def test_psi_interpretation_thresholds_ordered(self):
        assert _PSI_FRAGILE_THRESHOLD < _PSI_ROBUST_THRESHOLD

    def test_roi_interpretation_thresholds_ordered(self):
        assert _ROI_WEAK_THRESHOLD < _ROI_MODERATE_THRESHOLD

    def test_friction_interpretation_thresholds_ordered(self):
        assert _FRICTION_FAVORABLE_THRESHOLD < _FRICTION_MODERATE_THRESHOLD


# ============================================================================
# TEST: PROPIEDADES GLOBALES (INVARIANTES DEL MANIFOLD)
# ============================================================================


class TestManifoldInvariants:
    """Tests de propiedades que deben sostenerse para cualquier input válido.

    Estas son las 'leyes' del manifold que nunca deben violarse.
    """

    @pytest.fixture(params=[
        {"psi": 0.05, "roi": 0.0},
        {"psi": 0.5, "roi": 0.5},
        {"psi": 1.0, "roi": 1.0},
        {"psi": 2.0, "roi": 2.0},
        {"psi": 5.0, "roi": 5.0},
        {"psi": 0.05, "roi": 5.0},
        {"psi": 5.0, "roi": 0.0},
    ])
    def varied_inputs(self, request) -> OmegaInputs:
        return OmegaInputs(**request.param)

    def test_all_outputs_finite(self, manifold, varied_inputs):
        result = manifold._collapse(varied_inputs)
        metrics = result.metrics
        for f in [
            metrics.fragility_norm, metrics.roi_norm, metrics.misalignment,
            metrics.gravity_coupling, metrics.internal_tension,
            metrics.external_friction, metrics.anomaly_pressure,
            metrics.combinatorial_scale, metrics.friction_scale,
            metrics.improbability_lever, metrics.base_stress,
            metrics.fragility_penalty, metrics.total_stress,
            metrics.adjusted_stress,
        ]:
            assert math.isfinite(f)

    def test_verdict_always_valid(self, manifold, varied_inputs):
        result = manifold._collapse(varied_inputs)
        assert result.verdict in (
            VerdictLevel.VIABLE,
            VerdictLevel.CONDICIONAL,
            VerdictLevel.PRECAUCION,
            VerdictLevel.RECHAZAR,
        )

    def test_fragility_norm_always_unit_interval(self, manifold, varied_inputs):
        metrics = manifold._compute_metrics(varied_inputs)
        assert 0.0 <= metrics.fragility_norm <= 1.0

    def test_roi_norm_always_unit_interval(self, manifold, varied_inputs):
        metrics = manifold._compute_metrics(varied_inputs)
        assert 0.0 <= metrics.roi_norm <= 1.0

    def test_misalignment_always_unit_interval(self, manifold, varied_inputs):
        metrics = manifold._compute_metrics(varied_inputs)
        assert 0.0 <= metrics.misalignment <= 1.0

    def test_stress_always_nonnegative(self, manifold, varied_inputs):
        metrics = manifold._compute_metrics(varied_inputs)
        assert metrics.base_stress >= 0
        assert metrics.total_stress >= 0
        assert metrics.adjusted_stress >= 0

    def test_penalty_always_gte_one(self, manifold, varied_inputs):
        metrics = manifold._compute_metrics(varied_inputs)
        assert metrics.fragility_penalty >= 1.0

    def test_lever_always_clamped(self, manifold, varied_inputs):
        metrics = manifold._compute_metrics(varied_inputs)
        assert _IMPROBABILITY_CLAMP_LOW <= metrics.improbability_lever <= _IMPROBABILITY_CLAMP_HIGH

    def test_external_friction_always_gte_one(self, manifold, varied_inputs):
        metrics = manifold._compute_metrics(varied_inputs)
        assert metrics.external_friction >= 1.0


# ============================================================================
# TEST: REGRESIÓN
# ============================================================================


class TestRegressions:
    """Tests de regresión para bugs previamente identificados."""

    def test_misalignment_scale_symmetry(self, manifold):
        """Regresión: antes fragility y ROI estaban en escalas distintas,
        haciendo misalignment asimétricamente dominada por fragility."""
        # Con normalización, ambos en [0,1]: misalignment debe ser razonable
        inputs = OmegaInputs(psi=1.0, roi=1.0)
        metrics = manifold._compute_metrics(inputs)
        # fragility_norm ≈ 0.06, roi_norm = 0.2 → misalignment ≈ 0.14
        assert metrics.misalignment < 0.5, (
            f"Misalignment demasiado alto para inputs neutrales: {metrics.misalignment}"
        )

    def test_territory_empty_dict_consistency(self, manifold):
        """Regresión: territory_data={} producía fricciones calculadas
        pero territory_present=False, inconsistencia."""
        payload = {
            "territory_state": {},
        }
        inputs = OmegaInputs.from_payload(payload)
        assert inputs.territory_present is False
        # Las fricciones deben ser neutrales
        assert inputs.logistics_friction == 1.0
        assert inputs.social_friction == 1.0
        assert inputs.climate_entropy == 1.0

    def test_max_chars_includes_separator(self):
        """Regresión: get_active_context no contaba el \\n entre payloads."""
        sr = SynapticRegistry()
        sr.load_cartridge(
            ToonCartridge(name="a", domain="d", toon_payload="AAA", weight=2.0)
        )
        sr.load_cartridge(
            ToonCartridge(name="b", domain="d", toon_payload="BBB", weight=1.0)
        )
        # "AAA\nBBB" = 7 chars. Con max_chars=6, solo cabe "AAA"
        context = sr.get_active_context(max_chars=6)
        assert context == "AAA"

    def test_dominant_axis_deterministic(self, manifold):
        """Regresión: empates en riesgo dominante no eran deterministas."""
        metrics = OmegaMetrics(
            fragility_norm=0.5, roi_norm=0.3,
            misalignment=0.2, gravity_coupling=1.0,
            internal_tension=1.0, external_friction=2.0,
            anomaly_pressure=1.0, combinatorial_scale=1.0,
            friction_scale=1.0, improbability_lever=2.0,
            base_stress=2.0, fragility_penalty=2.0,
            total_stress=4.0, adjusted_stress=8.0,
        )
        results = set()
        for _ in range(100):
            axis, _ = manifold._identify_dominant_risk_axis(metrics)
            results.add(axis)
        assert len(results) == 1, f"No determinista: {results}"

    def test_bool_not_numeric(self):
        """Regresión: True/False podían interpretarse como 1/0."""
        assert _safe_float(True, 99.0) == 99.0
        assert _safe_float(False, 99.0) == 99.0
        assert _safe_int(True, 99) == 99
        assert _safe_int(False, 99) == 99

    def test_fragility_bounded_after_normalization(self, manifold):
        """Regresión: antes log1p(1/ε) producía ~13.8, mucho mayor que ROI max=5."""
        result = manifold._compute_fragility_normalized(_PSI_CLAMP_LOW)
        assert result == pytest.approx(1.0)
        result = manifold._compute_fragility_normalized(_PSI_CLAMP_HIGH)
        assert result < 0.1

    def test_payload_verdict_consistency(self, manifold, neutral_inputs):
        """Regresión: to_payload no incluía roi_norm ni fragility_norm."""
        result = manifold._collapse(neutral_inputs)
        payload = result.to_payload(synaptic_context_toon="ctx")
        assert "fragility_norm" in payload["omega_metrics"]
        assert "roi_norm" in payload["omega_metrics"]