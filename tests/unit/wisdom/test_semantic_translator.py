r"""
=============================================================================
Suite de Pruebas: Semantic Translator
Archivo: tests/test_semantic_translator.py
=============================================================================

Estrategia de verificación organizada por capas axiomáticas:

    Grupo 1: Invariantes físicos (Temperature, constantes)
    Grupo 2: Estructura algebraica (Retículos, homomorfismos)
    Grupo 3: Configuración y umbrales (Thresholds, validación)
    Grupo 4: Topología validada (Euler, Betti, Fiedler)
    Grupo 5: Difeomorfismo semántico (Mappers, narrativa causal)
    Grupo 6: Grafo y ciclos (GraphRAG, enumeración acotada)
    Grupo 7: Colapso del retículo (LatticeVerdictCollapse)
    Grupo 8: Caché de narrativas (LRU thread-safe)
    Grupo 9: Traductor principal (traducción por dominio)
    Grupo 10: Composición estratégica (reporte completo DIKW)
    Grupo 11: Funciones de utilidad y factory
    Grupo 12: Propiedades de regresión y casos extremos

Convenciones:
    - Cada test verifica una propiedad matemática o invariante específico
    - Los nombres siguen el patrón: test_<propiedad>_<condición>_<resultado>
    - Se usa pytest.mark.parametrize para cobertura combinatoria
=============================================================================
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import networkx as nx
import pytest

# --- Importaciones del módulo bajo prueba ---
from app.wisdom.semantic_translator import (
    # Constantes
    ABSOLUTE_ZERO_CELSIUS,
    KELVIN_OFFSET,
    EPSILON,
    MAX_SIMPLE_CYCLES_ENUMERATION,
    MAX_GRAPH_NODES_FOR_EIGENVECTOR,
    # Valor-objeto
    Temperature,
    # Excepciones
    SemanticTranslatorError,
    TopologyInvariantViolation,
    LatticeViolation,
    MetricsValidationError,
    GraphStructureError,
    # Configuración
    StabilityThresholds,
    TopologicalThresholds,
    ThermalThresholds,
    FinancialThresholds,
    TranslatorConfig,
    # Retículos
    VerdictLevel,
    SeverityLattice,
    SeverityToVerdictHomomorphism,
    # Difeomorfismo
    SemanticDiffeomorphismMapper,
    GraphRAGCausalNarrator,
    LatticeVerdictCollapse,
    # Veredicto financiero
    FinancialVerdict,
    # Estructuras validadas
    ValidatedTopology,
    HasBettiNumbers,
    # Resultados
    StratumAnalysisResult,
    StrategicReport,
    # Caché
    NarrativeCache,
    # Traductor principal
    SemanticTranslator,
    # Factory y utilidades
    create_translator,
    translate_metrics_to_narrative,
    verify_verdict_lattice,
    verify_severity_homomorphism,
)

try:
    from app.core.schemas import Stratum
except ImportError:
    from enum import IntEnum as _StratumBase

    class Stratum(_StratumBase):
        WISDOM = 0
        OMEGA = 1
        ALPHA = 2
        STRATEGY = 3
        TACTICS = 4
        PHYSICS = 5


# =========================================================================
# FIXTURES COMPARTIDOS
# =========================================================================
@pytest.fixture
def default_config() -> TranslatorConfig:
    """Configuración por defecto validada."""
    return TranslatorConfig()


@pytest.fixture
def mock_mic() -> MagicMock:
    """MICRegistry simulado que retorna narrativas deterministas."""
    mic = MagicMock()
    mic.project_intent.return_value = {
        "success": True,
        "narrative": "[TEST_NARRATIVE]",
    }
    return mic


@pytest.fixture
def translator(mock_mic: MagicMock) -> SemanticTranslator:
    """Traductor con MIC mockeado y caché habilitado."""
    return SemanticTranslator(mic=mock_mic, enable_cache=True)


@pytest.fixture
def translator_no_cache(mock_mic: MagicMock) -> SemanticTranslator:
    """Traductor sin caché."""
    return SemanticTranslator(mic=mock_mic, enable_cache=False)


@pytest.fixture
def valid_topology_connected() -> ValidatedTopology:
    """Topología conexa sin ciclos (β₀=1, β₁=0)."""
    return ValidatedTopology(
        beta_0=1, beta_1=0, beta_2=0,
        euler_characteristic=1,
        fiedler_value=1.0,
        spectral_gap=0.5,
        pyramid_stability=10.0,
        structural_entropy=0.2,
    )


@pytest.fixture
def valid_topology_with_cycles() -> ValidatedTopology:
    """Topología con ciclos (β₁=2)."""
    return ValidatedTopology(
        beta_0=1, beta_1=2, beta_2=0,
        euler_characteristic=-1,
        fiedler_value=0.8,
        spectral_gap=0.3,
        pyramid_stability=5.0,
        structural_entropy=0.4,
    )


@pytest.fixture
def valid_topology_fragmented() -> ValidatedTopology:
    """Topología fragmentada (β₀=3)."""
    return ValidatedTopology(
        beta_0=3, beta_1=0, beta_2=0,
        euler_characteristic=3,
        fiedler_value=0.0,
        spectral_gap=0.0,
        pyramid_stability=2.0,
        structural_entropy=0.6,
    )


@pytest.fixture
def simple_digraph_with_cycle() -> nx.DiGraph:
    """Grafo dirigido con un ciclo simple A→B→C→A."""
    G = nx.DiGraph()
    G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
    return G


@pytest.fixture
def simple_digraph_acyclic() -> nx.DiGraph:
    """Grafo dirigido acíclico (DAG)."""
    G = nx.DiGraph()
    G.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    return G


@pytest.fixture
def empty_digraph() -> nx.DiGraph:
    """Grafo dirigido vacío."""
    return nx.DiGraph()


@pytest.fixture
def viable_financial_metrics() -> Dict[str, Any]:
    """Métricas financieras viables."""
    return {
        "wacc": 0.08,
        "contingency": {"recommended": 0.10},
        "performance": {
            "recommendation": "ACEPTAR",
            "profitability_index": 1.5,
        },
    }


@pytest.fixture
def reject_financial_metrics() -> Dict[str, Any]:
    """Métricas financieras de rechazo."""
    return {
        "wacc": 0.20,
        "contingency": {"recommended": 0.25},
        "performance": {
            "recommendation": "RECHAZAR",
            "profitability_index": 0.5,
        },
    }


# =========================================================================
# GRUPO 1: INVARIANTES FÍSICOS — TEMPERATURE
# =========================================================================
class TestTemperature:
    """Verificación del valor-objeto Temperature."""

    def test_from_kelvin_valid(self):
        """Construcción desde Kelvin positivo."""
        t = Temperature.from_kelvin(300.0)
        assert t.kelvin == 300.0
        assert math.isclose(t.celsius, 300.0 - KELVIN_OFFSET, rel_tol=1e-9)

    def test_from_celsius_valid(self):
        """Construcción desde Celsius."""
        t = Temperature.from_celsius(25.0)
        expected_k = 25.0 + KELVIN_OFFSET
        assert math.isclose(t.kelvin, expected_k, rel_tol=1e-9)

    def test_absolute_zero_kelvin(self):
        """Cero absoluto desde Kelvin."""
        t = Temperature.from_kelvin(0.0)
        assert t.kelvin == 0.0
        assert t.is_absolute_zero

    def test_absolute_zero_celsius(self):
        """Cero absoluto desde Celsius."""
        t = Temperature.from_celsius(ABSOLUTE_ZERO_CELSIUS)
        assert t.is_absolute_zero

    def test_negative_kelvin_raises(self):
        """Kelvin negativo lanza ValueError."""
        with pytest.raises(ValueError, match="below absolute zero"):
            Temperature(kelvin=-1.0)

    def test_non_finite_raises(self):
        """Valores no finitos lanzan ValueError."""
        with pytest.raises(ValueError, match="finite"):
            Temperature(kelvin=float("inf"))
        with pytest.raises(ValueError, match="finite"):
            Temperature(kelvin=float("nan"))

    def test_epsilon_clamp_from_kelvin(self):
        """Clamp de valores negativos minúsculos a cero."""
        t = Temperature.from_kelvin(-1e-10)
        assert t.kelvin == 0.0

    def test_epsilon_clamp_from_celsius(self):
        """Clamp en conversión Celsius → Kelvin."""
        # Un valor de Celsius que produce Kelvin ligeramente negativo
        celsius = ABSOLUTE_ZERO_CELSIUS - 1e-10
        t = Temperature.from_celsius(celsius)
        assert t.kelvin == 0.0

    def test_is_absolute_zero_with_tolerance(self):
        """is_absolute_zero usa tolerancia ε."""
        t = Temperature.from_kelvin(EPSILON / 2)
        assert t.is_absolute_zero

    def test_is_absolute_zero_above_tolerance(self):
        """Valores por encima de ε no son cero absoluto."""
        t = Temperature.from_kelvin(EPSILON * 2)
        assert not t.is_absolute_zero

    def test_frozen_immutability(self):
        """El dataclass es inmutable (frozen=True)."""
        t = Temperature.from_kelvin(300.0)
        with pytest.raises(AttributeError):
            t.kelvin = 400.0  # type: ignore[misc]

    def test_order_total(self):
        """Orden total sobre Temperature."""
        t1 = Temperature.from_kelvin(100.0)
        t2 = Temperature.from_kelvin(200.0)
        t3 = Temperature.from_kelvin(200.0)
        assert t1 < t2
        assert t2 == t3
        assert t2 >= t1

    def test_str_repr(self):
        """Representación string y repr."""
        t = Temperature.from_kelvin(300.0)
        assert "°C" in str(t)
        assert "K" in str(t)
        assert "Temperature(kelvin=" in repr(t)


# =========================================================================
# GRUPO 2: ESTRUCTURA ALGEBRAICA — RETÍCULOS
# =========================================================================
class TestVerdictLevelLattice:
    """Verificación exhaustiva de las leyes del retículo VerdictLevel."""

    def test_all_lattice_laws_hold(self):
        """Verificación formal completa de las leyes del retículo."""
        results = VerdictLevel.verify_lattice_laws()
        for law_name, holds in results.items():
            assert holds, f"Ley del retículo violada: {law_name}"

    def test_bottom_is_viable(self):
        """⊥ = VIABLE."""
        assert VerdictLevel.bottom() == VerdictLevel.VIABLE

    def test_top_is_rechazar(self):
        """⊤ = RECHAZAR."""
        assert VerdictLevel.top() == VerdictLevel.RECHAZAR

    def test_supremum_empty_is_bottom(self):
        """⊔∅ = ⊥ (convención de retículos completos)."""
        assert VerdictLevel.supremum() == VerdictLevel.bottom()

    def test_infimum_empty_is_top(self):
        """⊓∅ = ⊤."""
        assert VerdictLevel.infimum() == VerdictLevel.top()

    @pytest.mark.parametrize("a,b,expected_join", [
        (VerdictLevel.VIABLE, VerdictLevel.RECHAZAR, VerdictLevel.RECHAZAR),
        (VerdictLevel.VIABLE, VerdictLevel.VIABLE, VerdictLevel.VIABLE),
        (VerdictLevel.CONDICIONAL, VerdictLevel.PRECAUCION, VerdictLevel.PRECAUCION),
        (VerdictLevel.REVISAR, VerdictLevel.REVISAR, VerdictLevel.REVISAR),
    ])
    def test_join_binary(self, a, b, expected_join):
        """Join binario a ⊔ b = max(a, b)."""
        assert (a | b) == expected_join
        assert a.join(b) == expected_join

    @pytest.mark.parametrize("a,b,expected_meet", [
        (VerdictLevel.VIABLE, VerdictLevel.RECHAZAR, VerdictLevel.VIABLE),
        (VerdictLevel.PRECAUCION, VerdictLevel.CONDICIONAL, VerdictLevel.CONDICIONAL),
    ])
    def test_meet_binary(self, a, b, expected_meet):
        """Meet binario a ⊓ b = min(a, b)."""
        assert (a & b) == expected_meet
        assert a.meet(b) == expected_meet

    def test_absorbing_element_join(self):
        """⊤ es absorbente para ⊔: a ⊔ ⊤ = ⊤ ∀a."""
        for a in VerdictLevel:
            assert (a | VerdictLevel.top()) == VerdictLevel.top()

    def test_absorbing_element_meet(self):
        """⊥ es absorbente para ⊓: a ⊓ ⊥ = ⊥ ∀a."""
        for a in VerdictLevel:
            assert (a & VerdictLevel.bottom()) == VerdictLevel.bottom()

    def test_operator_or_type_check(self):
        """Operador | retorna NotImplemented para tipos incompatibles."""
        result = VerdictLevel.VIABLE.__or__(42)
        assert result is NotImplemented

    def test_operator_and_type_check(self):
        """Operador & retorna NotImplemented para tipos incompatibles."""
        result = VerdictLevel.VIABLE.__and__("string")
        assert result is NotImplemented

    def test_supremum_variadic(self):
        """Supremum de múltiples elementos."""
        result = VerdictLevel.supremum(
            VerdictLevel.VIABLE,
            VerdictLevel.CONDICIONAL,
            VerdictLevel.PRECAUCION,
        )
        assert result == VerdictLevel.PRECAUCION

    def test_normalized_score_range(self):
        """Score normalizado ∈ [0, 1]."""
        for v in VerdictLevel:
            assert 0.0 <= v.normalized_score <= 1.0
        assert VerdictLevel.VIABLE.normalized_score == 0.0
        assert VerdictLevel.RECHAZAR.normalized_score == 1.0

    def test_emoji_all_defined(self):
        """Todos los niveles tienen emoji."""
        for v in VerdictLevel:
            assert len(v.emoji) > 0

    def test_description_all_defined(self):
        """Todos los niveles tienen descripción."""
        for v in VerdictLevel:
            assert len(v.description) > 0

    def test_is_positive_boundary(self):
        """is_positive solo para VIABLE y CONDICIONAL."""
        assert VerdictLevel.VIABLE.is_positive
        assert VerdictLevel.CONDICIONAL.is_positive
        assert not VerdictLevel.REVISAR.is_positive
        assert not VerdictLevel.PRECAUCION.is_positive
        assert not VerdictLevel.RECHAZAR.is_positive

    def test_is_negative_only_rechazar(self):
        """is_negative solo para RECHAZAR."""
        for v in VerdictLevel:
            if v == VerdictLevel.RECHAZAR:
                assert v.is_negative
            else:
                assert not v.is_negative

    def test_requires_attention(self):
        """requires_attention para PRECAUCION y RECHAZAR."""
        assert not VerdictLevel.VIABLE.requires_attention
        assert not VerdictLevel.CONDICIONAL.requires_attention
        assert not VerdictLevel.REVISAR.requires_attention
        assert VerdictLevel.PRECAUCION.requires_attention
        assert VerdictLevel.RECHAZAR.requires_attention

    def test_bounded_property(self):
        """Todos los elementos están acotados: ⊥ ≤ a ≤ ⊤."""
        results = VerdictLevel.verify_lattice_laws()
        assert results["bounded"]

    def test_total_order_property(self):
        """Orden total: ∀a,b, a ≤ b ∨ b ≤ a."""
        results = VerdictLevel.verify_lattice_laws()
        assert results["total_order"]


class TestSeverityLattice:
    """Verificación del retículo SeverityLattice."""

    def test_supremum_empty(self):
        """⊔∅ = ⊥ = VIABLE."""
        assert SeverityLattice.supremum() == SeverityLattice.VIABLE

    def test_infimum_empty(self):
        """⊓∅ = ⊤ = RECHAZAR."""
        assert SeverityLattice.infimum() == SeverityLattice.RECHAZAR

    def test_supremum_absorbing(self):
        """RECHAZAR es absorbente para ⊔."""
        assert SeverityLattice.supremum(
            SeverityLattice.VIABLE, SeverityLattice.RECHAZAR
        ) == SeverityLattice.RECHAZAR

    def test_infimum_absorbing(self):
        """VIABLE es absorbente para ⊓."""
        assert SeverityLattice.infimum(
            SeverityLattice.VIABLE, SeverityLattice.RECHAZAR
        ) == SeverityLattice.VIABLE

    def test_order_three_elements(self):
        """VIABLE < PRECAUCION < RECHAZAR."""
        assert SeverityLattice.VIABLE < SeverityLattice.PRECAUCION
        assert SeverityLattice.PRECAUCION < SeverityLattice.RECHAZAR


class TestSeverityToVerdictHomomorphism:
    """Verificación del homomorfismo φ: SeverityLattice → VerdictLevel."""

    def test_homomorphism_is_valid(self):
        """El homomorfismo preserva ⊔ y ⊓."""
        assert SeverityToVerdictHomomorphism.verify_homomorphism()

    def test_preserves_bottom(self):
        """φ(⊥_S) = ⊥_V."""
        assert SeverityToVerdictHomomorphism.apply(
            SeverityLattice.VIABLE
        ) == VerdictLevel.VIABLE

    def test_preserves_top(self):
        """φ(⊤_S) = ⊤_V."""
        assert SeverityToVerdictHomomorphism.apply(
            SeverityLattice.RECHAZAR
        ) == VerdictLevel.RECHAZAR

    def test_injective(self):
        """El homomorfismo es inyectivo (mapeos distintos)."""
        images = set()
        for sev in SeverityLattice:
            img = SeverityToVerdictHomomorphism.apply(sev)
            assert img not in images, f"φ no es inyectivo en {sev}"
            images.add(img)

    def test_order_preserving(self):
        """φ preserva el orden: a ≤ b ⟹ φ(a) ≤ φ(b)."""
        for a in SeverityLattice:
            for b in SeverityLattice:
                if a <= b:
                    fa = SeverityToVerdictHomomorphism.apply(a)
                    fb = SeverityToVerdictHomomorphism.apply(b)
                    assert fa <= fb


# =========================================================================
# GRUPO 3: CONFIGURACIÓN Y UMBRALES
# =========================================================================
class TestStabilityThresholds:
    """Verificación de StabilityThresholds."""

    def test_default_invariant(self):
        """Invariante: 0 ≤ critical < warning < solid."""
        st = StabilityThresholds()
        assert 0 <= st.critical < st.warning < st.solid

    def test_invalid_ordering_raises(self):
        """Orden incorrecto lanza ValueError."""
        with pytest.raises(ValueError, match="strict ordering"):
            StabilityThresholds(critical=5.0, warning=3.0, solid=10.0)

    def test_negative_critical_raises(self):
        """Critical negativo lanza ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            StabilityThresholds(critical=-1.0)

    def test_non_finite_raises(self):
        """Valores no finitos lanzan ValueError."""
        with pytest.raises(ValueError, match="finite"):
            StabilityThresholds(critical=float("inf"))

    @pytest.mark.parametrize("stability,expected", [
        (-1.0, "invalid"),
        (float("nan"), "invalid"),
        (0.5, "critical"),
        (2.0, "warning"),
        (5.0, "stable"),
        (15.0, "robust"),
    ])
    def test_classify(self, stability, expected):
        """Clasificación de estabilidad."""
        st = StabilityThresholds()
        assert st.classify(stability) == expected

    @pytest.mark.parametrize("stability,expected_score", [
        (-1.0, 1.0),
        (0.0, 1.0),
        (5.0, 0.5),
        (10.0, 0.0),
        (20.0, 0.0),
        (float("nan"), 1.0),
    ])
    def test_severity_score(self, stability, expected_score):
        """Score de severidad normalizado."""
        st = StabilityThresholds()
        score = st.severity_score(stability)
        assert math.isclose(score, expected_score, abs_tol=1e-9)
        assert 0.0 <= score <= 1.0


class TestTopologicalThresholds:
    """Verificación de TopologicalThresholds."""

    def test_default_invariants(self):
        """Invariantes por defecto."""
        tt = TopologicalThresholds()
        assert tt.connected_components_optimal >= 1
        assert 0 <= tt.cycles_optimal <= tt.cycles_warning <= tt.cycles_critical
        assert 0 <= tt.fiedler_connected_threshold < tt.fiedler_robust_threshold

    @pytest.mark.parametrize("beta_0,expected", [
        (0, "empty"),
        (1, "unified"),
        (3, "fragmented"),
        (10, "severely_fragmented"),
    ])
    def test_classify_connectivity(self, beta_0, expected):
        tt = TopologicalThresholds()
        assert tt.classify_connectivity(beta_0) == expected

    @pytest.mark.parametrize("beta_1,expected", [
        (-1, "invalid"),
        (0, "clean"),
        (1, "minor"),
        (2, "moderate"),
        (5, "critical"),
    ])
    def test_classify_cycles(self, beta_1, expected):
        tt = TopologicalThresholds()
        assert tt.classify_cycles(beta_1) == expected

    @pytest.mark.parametrize("fiedler,expected", [
        (float("nan"), "invalid"),
        (-0.01, "disconnected"),
        (0.0, "disconnected"),
        (1e-5, "weakly_connected"),
        (1.0, "strongly_connected"),
    ])
    def test_classify_spectral_connectivity(self, fiedler, expected):
        tt = TopologicalThresholds()
        assert tt.classify_spectral_connectivity(fiedler) == expected

    def test_validate_euler_characteristic_valid(self):
        """χ = β₀ − β₁ + β₂ → True."""
        assert TopologicalThresholds.validate_euler_characteristic(
            beta_0=1, beta_1=0, beta_2=0, chi=1
        )

    def test_validate_euler_characteristic_invalid(self):
        """χ ≠ β₀ − β₁ + β₂ → False."""
        assert not TopologicalThresholds.validate_euler_characteristic(
            beta_0=1, beta_1=0, beta_2=0, chi=5
        )


class TestThermalThresholds:
    """Verificación de ThermalThresholds."""

    def test_default_invariants(self):
        """Invariantes de umbrales térmicos."""
        th = ThermalThresholds()
        assert (th.temperature_cold < th.temperature_warm
                < th.temperature_hot < th.temperature_critical)
        assert 0 <= th.entropy_low < th.entropy_high < th.entropy_death <= 1
        assert 0 <= th.exergy_poor < th.exergy_efficient <= 1
        assert th.heat_capacity_minimum > 0

    @pytest.mark.parametrize("temp,expected", [
        (float("nan"), "invalid"),
        (-300.0, "invalid"),
        (10.0, "cold"),
        (25.0, "stable"),
        (40.0, "warm"),
        (60.0, "hot"),
        (80.0, "critical"),
    ])
    def test_classify_temperature(self, temp, expected):
        th = ThermalThresholds()
        assert th.classify_temperature(temp) == expected

    @pytest.mark.parametrize("entropy,expected", [
        (-0.1, "invalid"),
        (1.1, "invalid"),
        (float("nan"), "invalid"),
        (0.1, "low"),
        (0.5, "moderate"),
        (0.8, "high"),
        (0.96, "death"),
    ])
    def test_classify_entropy(self, entropy, expected):
        th = ThermalThresholds()
        assert th.classify_entropy(entropy) == expected

    @pytest.mark.parametrize("exergy,expected", [
        (float("nan"), "invalid"),
        (-0.1, "invalid"),
        (0.2, "poor"),
        (0.5, "moderate"),
        (0.8, "efficient"),
    ])
    def test_classify_exergy(self, exergy, expected):
        th = ThermalThresholds()
        assert th.classify_exergy(exergy) == expected

    def test_invalid_entropy_ordering_raises(self):
        with pytest.raises(ValueError, match="Entropy"):
            ThermalThresholds(entropy_low=0.8, entropy_high=0.3)


class TestFinancialThresholds:
    """Verificación de FinancialThresholds."""

    def test_default_invariants(self):
        ft = FinancialThresholds()
        assert 0 < ft.wacc_low < ft.wacc_moderate < ft.wacc_high
        assert (0 < ft.profitability_marginal < ft.profitability_good
                < ft.profitability_excellent)

    @pytest.mark.parametrize("pi,expected", [
        (float("nan"), "invalid"),
        (0.5, "poor"),
        (1.0, "marginal"),
        (1.3, "good"),
        (2.0, "excellent"),
    ])
    def test_classify_profitability(self, pi, expected):
        ft = FinancialThresholds()
        assert ft.classify_profitability(pi) == expected


class TestTranslatorConfig:
    """Verificación de TranslatorConfig."""

    def test_default_construction(self):
        """Construcción por defecto exitosa."""
        config = TranslatorConfig()
        assert config.max_cycle_path_display >= 1
        assert config.max_narrative_length >= 100

    def test_invalid_max_cycle_display(self):
        with pytest.raises(ValueError, match="max_cycle_path_display"):
            TranslatorConfig(max_cycle_path_display=0)

    def test_invalid_max_narrative_length(self):
        with pytest.raises(ValueError, match="max_narrative_length"):
            TranslatorConfig(max_narrative_length=50)


# =========================================================================
# GRUPO 4: TOPOLOGÍA VALIDADA
# =========================================================================
class TestValidatedTopology:
    """Verificación de ValidatedTopology."""

    def test_valid_construction(self, valid_topology_connected):
        """Construcción válida con invariante de Euler."""
        t = valid_topology_connected
        assert t.euler_characteristic == t.beta_0 - t.beta_1 + t.beta_2

    def test_euler_invariant_violation_raises(self):
        """Violación de χ = β₀ − β₁ + β₂ lanza excepción."""
        with pytest.raises(TopologyInvariantViolation):
            ValidatedTopology(
                beta_0=1, beta_1=0, beta_2=0,
                euler_characteristic=99,
                fiedler_value=1.0, spectral_gap=0.0,
                pyramid_stability=1.0, structural_entropy=0.0,
            )

    def test_negative_betti_raises(self):
        """Números de Betti negativos lanzan ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            ValidatedTopology(
                beta_0=-1, beta_1=0, beta_2=0,
                euler_characteristic=-1,
                fiedler_value=1.0, spectral_gap=0.0,
                pyramid_stability=1.0, structural_entropy=0.0,
            )

    def test_negative_fiedler_raises(self):
        """Fiedler significativamente negativo lanza ValueError."""
        with pytest.raises(ValueError, match="Fiedler"):
            ValidatedTopology(
                beta_0=1, beta_1=0, beta_2=0,
                euler_characteristic=1,
                fiedler_value=-1.0, spectral_gap=0.0,
                pyramid_stability=1.0, structural_entropy=0.0,
            )

    def test_from_metrics_dict_auto_correct(self):
        """from_metrics corrige Euler automáticamente (strict=False)."""
        metrics = {
            "beta_0": 2, "beta_1": 1, "beta_2": 0,
            "euler_characteristic": 999,
            "fiedler_value": 0.5,
            "spectral_gap": 0.1,
            "pyramid_stability": 3.0,
            "structural_entropy": 0.3,
        }
        topo = ValidatedTopology.from_metrics(metrics, strict=False)
        assert topo.euler_characteristic == 2 - 1 + 0  # = 1

    def test_from_metrics_dict_strict_raises(self):
        """from_metrics con strict=True lanza excepción."""
        metrics = {
            "beta_0": 1, "beta_1": 0, "beta_2": 0,
            "euler_characteristic": 42,
        }
        with pytest.raises(TopologyInvariantViolation):
            ValidatedTopology.from_metrics(metrics, strict=True)

    def test_from_metrics_sanitizes_nan(self):
        """from_metrics sanitiza NaN a 0.0."""
        metrics = {
            "beta_0": 1, "beta_1": 0, "beta_2": 0,
            "fiedler_value": float("nan"),
            "spectral_gap": float("inf"),
            "pyramid_stability": float("-inf"),
            "structural_entropy": float("nan"),
        }
        topo = ValidatedTopology.from_metrics(metrics, strict=False)
        assert topo.fiedler_value == 0.0
        assert topo.spectral_gap == 0.0
        assert topo.pyramid_stability == 0.0

    def test_is_connected(self, valid_topology_connected):
        assert valid_topology_connected.is_connected

    def test_not_connected(self, valid_topology_fragmented):
        assert not valid_topology_fragmented.is_connected

    def test_has_cycles(self, valid_topology_with_cycles):
        assert valid_topology_with_cycles.has_cycles

    def test_no_cycles(self, valid_topology_connected):
        assert not valid_topology_connected.has_cycles

    def test_genus(self, valid_topology_with_cycles):
        """genus := β₁ para 1-complejos."""
        assert valid_topology_with_cycles.genus == 2

    def test_genus_surface(self, valid_topology_with_cycles):
        """genus_surface := β₁ / 2 para superficies cerradas."""
        assert valid_topology_with_cycles.genus_surface == 1.0

    def test_is_spectrally_connected(self, valid_topology_connected):
        assert valid_topology_connected.is_spectrally_connected

    def test_not_spectrally_connected(self, valid_topology_fragmented):
        """λ₂ = 0 ⟹ no espectralmente conexo."""
        assert not valid_topology_fragmented.is_spectrally_connected

    def test_topology_invariant_violation_attributes(self):
        """TopologyInvariantViolation almacena datos correctos."""
        try:
            ValidatedTopology(
                beta_0=2, beta_1=0, beta_2=0,
                euler_characteristic=5,
                fiedler_value=1.0, spectral_gap=0.0,
                pyramid_stability=1.0, structural_entropy=0.0,
            )
            pytest.fail("Expected TopologyInvariantViolation")
        except TopologyInvariantViolation as e:
            assert e.beta_0 == 2
            assert e.beta_1 == 0
            assert e.beta_2 == 0
            assert e.chi == 5
            assert e.expected_chi == 2


# =========================================================================
# GRUPO 5: DIFEOMORFISMO SEMÁNTICO
# =========================================================================
class TestSemanticDiffeomorphismMapper:
    """Verificación de los mapeos del Funtor F."""

    def test_map_betti_1_cycles_empty(self):
        """Lista vacía retorna string vacío."""
        assert SemanticDiffeomorphismMapper.map_betti_1_cycles([]) == ""

    def test_map_betti_1_cycles_single_node(self):
        """Un solo nodo retorna string vacío."""
        assert SemanticDiffeomorphismMapper.map_betti_1_cycles(["A"]) == ""

    def test_map_betti_1_cycles_valid(self):
        """Ciclo válido genera narrativa con cadena patológica."""
        result = SemanticDiffeomorphismMapper.map_betti_1_cycles(
            ["A", "B", "C"]
        )
        assert "VETO ESTRUCTURAL" in result
        assert "A ➔ B ➔ C ➔ A" in result

    def test_map_pyramid_instability(self):
        """Mapeo de Pirámide Invertida genera alerta."""
        result = SemanticDiffeomorphismMapper.map_pyramid_instability(
            0.5, "Cemento XYZ"
        )
        assert "ALERTA DE QUIEBRA" in result
        assert "Cemento XYZ" in result
        assert "0.50" in result

    def test_map_betti_0_fragmentation(self):
        """Mapeo de Fragmentación genera alerta con costo."""
        result = SemanticDiffeomorphismMapper.map_betti_0_fragmentation(
            3, 50000.0
        )
        assert "FUGA DE CAPITAL" in result
        assert "3" in result
        assert "$50,000.00" in result


# =========================================================================
# GRUPO 6: GRAFO Y CICLOS
# =========================================================================
class TestGraphRAGCausalNarrator:
    """Verificación del narrador causal basado en grafos."""

    def test_invalid_graph_type_raises(self):
        """Tipo de grafo incorrecto lanza GraphStructureError."""
        with pytest.raises(GraphStructureError):
            GraphRAGCausalNarrator("not_a_graph")

    def test_narrate_nominal(self, simple_digraph_acyclic):
        """Grafo acíclico produce estado nominal."""
        narrator = GraphRAGCausalNarrator(simple_digraph_acyclic)
        result = narrator.narrate_topological_collapse(betti_1=0, psi=5.0)
        assert "ESTADO NOMINAL" in result

    def test_narrate_with_cycles(self, simple_digraph_with_cycle):
        """Grafo con ciclos produce veto estructural."""
        narrator = GraphRAGCausalNarrator(simple_digraph_with_cycle)
        result = narrator.narrate_topological_collapse(betti_1=1, psi=5.0)
        assert "VETO ESTRUCTURAL" in result

    def test_narrate_with_instability(self, simple_digraph_acyclic):
        """Ψ < 1.0 produce alerta de quiebra."""
        narrator = GraphRAGCausalNarrator(simple_digraph_acyclic)
        result = narrator.narrate_topological_collapse(betti_1=0, psi=0.5)
        assert "ALERTA DE QUIEBRA" in result

    def test_extract_causality_acyclic(self, simple_digraph_acyclic):
        """Grafo acíclico no tiene ciclos para extraer."""
        narrator = GraphRAGCausalNarrator(simple_digraph_acyclic)
        cycles = narrator.extract_causality()
        assert cycles == []

    def test_extract_causality_with_cycle(self, simple_digraph_with_cycle):
        """Grafo con ciclo retorna al menos un ciclo."""
        narrator = GraphRAGCausalNarrator(simple_digraph_with_cycle)
        cycles = narrator.extract_causality()
        assert len(cycles) >= 1
        # Cada ciclo debe tener al menos 2 nodos
        for c in cycles:
            assert len(c) >= 2

    def test_enumerate_cycles_bounded(self):
        """Enumeración respeta límite máximo."""
        G = nx.DiGraph()
        # Crear grafo completo dirigido (muchos ciclos)
        for i in range(5):
            for j in range(5):
                if i != j:
                    G.add_edge(str(i), str(j))
        narrator = GraphRAGCausalNarrator(G, max_cycles=3)
        cycles = narrator._enumerate_cycles_bounded()
        assert len(cycles) <= 3

    def test_enumerate_cycles_empty_graph(self, empty_digraph):
        """Grafo vacío retorna lista vacía."""
        narrator = GraphRAGCausalNarrator(empty_digraph)
        assert narrator._enumerate_cycles_bounded() == []

    def test_find_critical_node_empty(self, empty_digraph):
        """Grafo vacío retorna nodo desconocido."""
        narrator = GraphRAGCausalNarrator(empty_digraph)
        assert "Desconocido" in narrator._find_critical_node()

    def test_find_critical_node_strongly_connected(self):
        """Grafo fuertemente conexo usa eigenvector centrality."""
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
        narrator = GraphRAGCausalNarrator(G)
        node = narrator._find_critical_node()
        assert node in ["A", "B", "C"]

    def test_find_critical_node_not_strongly_connected(self):
        """Grafo no fuertemente conexo usa degree centrality."""
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("C", "D")])
        narrator = GraphRAGCausalNarrator(G)
        node = narrator._find_critical_node()
        assert node in ["A", "B", "C", "D"]

    def test_find_critical_node_by_degree(self, simple_digraph_acyclic):
        """Fallback por grado identifica nodo más conectado."""
        narrator = GraphRAGCausalNarrator(simple_digraph_acyclic)
        node = narrator._find_critical_node_by_degree()
        # D tiene in_degree=2 y out_degree=0, A tiene in=0, out=2
        assert node in ["A", "D"]

    def test_deduplicate_cycles_by_support(self):
        """Deduplicación por soporte elimina ciclos homólogos."""
        G = nx.DiGraph()
        G.add_edges_from([
            ("A", "B"), ("B", "C"), ("C", "A"),
            ("B", "D"), ("D", "C"),
        ])
        narrator = GraphRAGCausalNarrator(G)
        cycles = [["A", "B", "C"], ["A", "B", "D", "C"]]
        unique = narrator._deduplicate_cycles_by_support(cycles)
        # Deben ser 1 o 2 dependiendo del soporte compartido
        assert 1 <= len(unique) <= 2


# =========================================================================
# GRUPO 7: COLAPSO DEL RETÍCULO
# =========================================================================
class TestLatticeVerdictCollapse:
    """Verificación del motor determinista de colapso."""

    def test_supremum_both_viable(self):
        """VIABLE ⊔ VIABLE = VIABLE."""
        result = LatticeVerdictCollapse.compute_supremum(
            SeverityLattice.VIABLE, SeverityLattice.VIABLE
        )
        assert result == SeverityLattice.VIABLE

    def test_supremum_viable_rechazar(self):
        """VIABLE ⊔ RECHAZAR = RECHAZAR (⊤ absorbe)."""
        result = LatticeVerdictCollapse.compute_supremum(
            SeverityLattice.VIABLE, SeverityLattice.RECHAZAR
        )
        assert result == SeverityLattice.RECHAZAR

    def test_supremum_commutative(self):
        """Conmutatividad: f ⊔ t = t ⊔ f."""
        for f in SeverityLattice:
            for t in SeverityLattice:
                assert (
                    LatticeVerdictCollapse.compute_supremum(f, t)
                    == LatticeVerdictCollapse.compute_supremum(t, f)
                )

    def test_enforce_fast_fail_topology_reject(
        self, simple_digraph_with_cycle,
    ):
        """Veto topológico produce RECHAZAR independiente de finanzas."""
        result = LatticeVerdictCollapse.enforce_semantic_diffeomorphism(
            roi_viable=True,
            betti_1=1,
            psi=5.0,
            graph=simple_digraph_with_cycle,
        )
        assert "RECHAZAR" in result

    def test_enforce_fast_fail_instability(self, simple_digraph_acyclic):
        """Ψ < 1.0 produce RECHAZAR."""
        result = LatticeVerdictCollapse.enforce_semantic_diffeomorphism(
            roi_viable=True,
            betti_1=0,
            psi=0.5,
            graph=simple_digraph_acyclic,
        )
        assert "RECHAZAR" in result

    def test_enforce_nominal_state(self, simple_digraph_acyclic):
        """Estado nominal (β₁=0, Ψ≥1, ROI ok) no produce RECHAZAR."""
        result = LatticeVerdictCollapse.enforce_semantic_diffeomorphism(
            roi_viable=True,
            betti_1=0,
            psi=5.0,
            graph=simple_digraph_acyclic,
        )
        assert "RECHAZAR" not in result or "VIABLE" in result

    def test_enforce_fast_fail_justification(
        self, simple_digraph_with_cycle,
    ):
        """Fast-fail incluye justificación del cortocircuito absorbente."""
        result = LatticeVerdictCollapse.enforce_semantic_diffeomorphism(
            roi_viable=True,
            betti_1=2,
            psi=0.5,
            graph=simple_digraph_with_cycle,
        )
        assert "cortocircuito absorbente" in result
        assert "a ⊔ ⊤ = ⊤" in result


# =========================================================================
# GRUPO 8: CACHÉ DE NARRATIVAS
# =========================================================================
class TestNarrativeCache:
    """Verificación del caché LRU thread-safe."""

    def test_construction_valid(self):
        cache = NarrativeCache(maxsize=10)
        assert cache.stats["maxsize"] == 10
        assert cache.stats["size"] == 0

    def test_construction_invalid_maxsize(self):
        with pytest.raises(ValueError, match="maxsize"):
            NarrativeCache(maxsize=0)

    def test_put_and_get(self):
        cache = NarrativeCache(maxsize=10)
        cache.put("domain", "class", {"key": "val"}, "narrative_text")
        result = cache.get("domain", "class", {"key": "val"})
        assert result == "narrative_text"

    def test_get_miss(self):
        cache = NarrativeCache(maxsize=10)
        result = cache.get("unknown", "domain", {})
        assert result is None

    def test_lru_eviction(self):
        """Evicción LRU cuando se excede maxsize."""
        cache = NarrativeCache(maxsize=2)
        cache.put("d", "c1", {}, "first")
        cache.put("d", "c2", {}, "second")
        cache.put("d", "c3", {}, "third")  # Evicta "first"
        assert cache.get("d", "c1", {}) is None
        assert cache.get("d", "c2", {}) is not None
        assert cache.get("d", "c3", {}) is not None

    def test_lru_access_updates_order(self):
        """Acceso actualiza posición LRU."""
        cache = NarrativeCache(maxsize=2)
        cache.put("d", "c1", {}, "first")
        cache.put("d", "c2", {}, "second")
        # Acceder c1 lo mueve al final
        cache.get("d", "c1", {})
        cache.put("d", "c3", {}, "third")  # Evicta c2 (más antiguo)
        assert cache.get("d", "c1", {}) == "first"
        assert cache.get("d", "c2", {}) is None

    def test_clear(self):
        cache = NarrativeCache(maxsize=10)
        cache.put("d", "c", {}, "text")
        cache.clear()
        assert cache.stats["size"] == 0
        assert cache.stats["hits"] == 0
        assert cache.stats["misses"] == 0

    def test_stats_tracking(self):
        cache = NarrativeCache(maxsize=10)
        cache.put("d", "c", {}, "text")
        cache.get("d", "c", {})  # hit
        cache.get("d", "unknown", {})  # miss
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1

    def test_deterministic_keys(self):
        """Claves SHA-256 son deterministas."""
        key1 = NarrativeCache._make_key("d", "c", {"a": 1})
        key2 = NarrativeCache._make_key("d", "c", {"a": 1})
        assert key1 == key2
        assert len(key1) == 64  # SHA-256 hex

    def test_different_params_different_keys(self):
        key1 = NarrativeCache._make_key("d", "c", {"a": 1})
        key2 = NarrativeCache._make_key("d", "c", {"a": 2})
        assert key1 != key2

    def test_thread_safety(self):
        """Operaciones concurrentes no causan corrupción."""
        cache = NarrativeCache(maxsize=100)
        errors: List[str] = []

        def writer(tid: int):
            try:
                for i in range(50):
                    cache.put("d", f"c_{tid}_{i}", {}, f"v_{tid}_{i}")
            except Exception as e:
                errors.append(str(e))

        def reader(tid: int):
            try:
                for i in range(50):
                    cache.get("d", f"c_{tid}_{i}", {})
            except Exception as e:
                errors.append(str(e))

        threads = []
        for tid in range(4):
            threads.append(threading.Thread(target=writer, args=(tid,)))
            threads.append(threading.Thread(target=reader, args=(tid,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =========================================================================
# GRUPO 9: FINANCIAL VERDICT
# =========================================================================
class TestFinancialVerdict:
    """Verificación del FinancialVerdict."""

    @pytest.mark.parametrize("string_input,expected", [
        ("ACEPTAR", FinancialVerdict.ACCEPT),
        ("ACCEPT", FinancialVerdict.ACCEPT),
        ("OK", FinancialVerdict.ACCEPT),
        ("CONDICIONAL", FinancialVerdict.CONDITIONAL),
        ("REVISAR", FinancialVerdict.REVIEW),
        ("RECHAZAR", FinancialVerdict.REJECT),
        ("REJECT", FinancialVerdict.REJECT),
        ("NO", FinancialVerdict.REJECT),
        ("unknown_value", FinancialVerdict.REVIEW),
        ("", FinancialVerdict.REVIEW),
    ])
    def test_from_string(self, string_input, expected):
        assert FinancialVerdict.from_string(string_input) == expected

    def test_from_string_none(self):
        assert FinancialVerdict.from_string(None) == FinancialVerdict.REVIEW

    def test_from_string_non_string(self):
        assert FinancialVerdict.from_string(42) == FinancialVerdict.REVIEW

    def test_to_verdict_level_mapping(self):
        """Homomorfismo al retículo de veredictos."""
        assert (FinancialVerdict.ACCEPT.to_verdict_level()
                == VerdictLevel.VIABLE)
        assert (FinancialVerdict.CONDITIONAL.to_verdict_level()
                == VerdictLevel.CONDICIONAL)
        assert (FinancialVerdict.REVIEW.to_verdict_level()
                == VerdictLevel.REVISAR)
        assert (FinancialVerdict.REJECT.to_verdict_level()
                == VerdictLevel.RECHAZAR)

    def test_case_insensitive(self):
        assert (FinancialVerdict.from_string("aceptar")
                == FinancialVerdict.ACCEPT)
        assert (FinancialVerdict.from_string("  RECHAZAR  ")
                == FinancialVerdict.REJECT)


# =========================================================================
# GRUPO 10: TRADUCTOR PRINCIPAL — TRADUCCIÓN POR DOMINIO
# =========================================================================
class TestSemanticTranslatorTopology:
    """Verificación de traducción topológica."""

    def test_translate_topology_connected_stable(
        self, translator, valid_topology_connected,
    ):
        """Topología conexa y estable produce veredicto positivo."""
        narrative, verdict = translator.translate_topology(
            valid_topology_connected, stability=10.0,
        )
        assert isinstance(narrative, str)
        assert len(narrative) > 0
        assert verdict.is_positive or verdict == VerdictLevel.CONDICIONAL

    def test_translate_topology_with_cycles(
        self, translator, valid_topology_with_cycles,
    ):
        """Topología con ciclos produce veredicto severo."""
        narrative, verdict = translator.translate_topology(
            valid_topology_with_cycles, stability=5.0,
        )
        assert verdict.value >= VerdictLevel.PRECAUCION.value

    def test_translate_topology_fragmented(
        self, translator, valid_topology_fragmented,
    ):
        """Topología fragmentada produce veredicto no-viable."""
        narrative, verdict = translator.translate_topology(
            valid_topology_fragmented, stability=2.0,
        )
        assert verdict.value >= VerdictLevel.CONDICIONAL.value

    def test_translate_topology_invalid_stability_raises(
        self, translator, valid_topology_connected,
    ):
        """Estabilidad negativa lanza MetricsValidationError."""
        with pytest.raises(MetricsValidationError):
            translator.translate_topology(
                valid_topology_connected, stability=-1.0,
            )

    def test_translate_topology_nan_stability_raises(
        self, translator, valid_topology_connected,
    ):
        with pytest.raises(MetricsValidationError):
            translator.translate_topology(
                valid_topology_connected, stability=float("nan"),
            )

    def test_translate_topology_from_dict(self, translator):
        """Acepta diccionario como métricas."""
        metrics = {
            "beta_0": 1, "beta_1": 0, "beta_2": 0,
            "fiedler_value": 1.0, "spectral_gap": 0.5,
            "pyramid_stability": 10.0, "structural_entropy": 0.1,
        }
        narrative, verdict = translator.translate_topology(
            metrics, stability=10.0,
        )
        assert isinstance(narrative, str)

    def test_translate_topology_synergy_risk(
        self, translator, valid_topology_connected,
    ):
        """Sinergia de riesgo produce RECHAZAR."""
        synergy = {"synergy_detected": True, "intersecting_cycles_count": 2}
        _, verdict = translator.translate_topology(
            valid_topology_connected, stability=10.0,
            synergy_risk=synergy,
        )
        assert verdict == VerdictLevel.RECHAZAR

    def test_translate_topology_spectral_resonance(
        self, translator, valid_topology_connected,
    ):
        """Resonancia espectral produce al menos PRECAUCION."""
        spectral = {"resonance_risk": True, "wavelength": 0.5}
        _, verdict = translator.translate_topology(
            valid_topology_connected, stability=10.0,
            spectral=spectral,
        )
        assert verdict.value >= VerdictLevel.PRECAUCION.value


class TestSemanticTranslatorThermodynamics:
    """Verificación de traducción termodinámica."""

    def test_translate_cold_system(self, translator):
        """Sistema frío produce veredicto positivo."""
        metrics = {
            "system_temperature": 15.0,
            "entropy": 0.1,
            "heat_capacity": 0.5,
        }
        _, verdict = translator.translate_thermodynamics(metrics)
        assert verdict.is_positive

    def test_translate_critical_temperature(self, translator):
        """Temperatura crítica produce RECHAZAR."""
        metrics = {
            "system_temperature": 80.0,
            "entropy": 0.1,
            "heat_capacity": 0.5,
        }
        _, verdict = translator.translate_thermodynamics(
            metrics, fiedler_value=0.0001,
        )
        # Con Fiedler bajo, es bottleneck → RECHAZAR
        assert verdict == VerdictLevel.RECHAZAR

    def test_translate_thermal_death(self, translator):
        """Entropía ≥ death produce RECHAZAR."""
        metrics = {
            "system_temperature": 25.0,
            "entropy": 0.99,
            "heat_capacity": 0.5,
        }
        _, verdict = translator.translate_thermodynamics(metrics)
        assert verdict == VerdictLevel.RECHAZAR

    def test_thermal_bottleneck_low_fiedler(self, translator):
        """Fiedler < threshold produce embotellamiento térmico."""
        metrics = {
            "system_temperature": 50.0,
            "entropy": 0.3,
            "heat_capacity": 0.5,
        }
        narrative, verdict = translator.translate_thermodynamics(
            metrics, fiedler_value=1e-8,
        )
        assert "Embotellamiento" in narrative
        assert verdict == VerdictLevel.RECHAZAR

    def test_low_heat_capacity_warning(self, translator):
        """Baja inercia financiera produce PRECAUCION."""
        metrics = {
            "system_temperature": 25.0,
            "entropy": 0.3,
            "heat_capacity": 0.05,
        }
        narrative, _ = translator.translate_thermodynamics(metrics)
        assert "Hoja al Viento" in narrative


class TestSemanticTranslatorFinancial:
    """Verificación de traducción financiera."""

    def test_translate_viable(self, translator, viable_financial_metrics):
        _, verdict, fin = translator.translate_financial(
            viable_financial_metrics
        )
        assert verdict == VerdictLevel.VIABLE
        assert fin == FinancialVerdict.ACCEPT

    def test_translate_reject(self, translator, reject_financial_metrics):
        _, verdict, fin = translator.translate_financial(
            reject_financial_metrics
        )
        assert verdict == VerdictLevel.RECHAZAR
        assert fin == FinancialVerdict.REJECT

    def test_translate_invalid_type_raises(self, translator):
        with pytest.raises(MetricsValidationError):
            translator.translate_financial("not_a_dict")

    def test_translate_empty_metrics(self, translator):
        """Métricas vacías producen REVIEW por defecto."""
        _, verdict, fin = translator.translate_financial({})
        assert fin == FinancialVerdict.REVIEW


# =========================================================================
# GRUPO 11: MAHALANOBIS Y HELPERS
# =========================================================================
class TestMahalanobisRetrieval:
    """Verificación de la recuperación Mahalanobis."""

    def test_empty_candidates(self, translator):
        result = translator._mahalanobis_retrieve(
            np.array([1.0, 0.0]),
            np.eye(2),
            [],
        )
        assert result == ""

    def test_identity_metric(self, translator):
        """Con G = I, equivale a distancia euclidiana."""
        query = np.array([1.0, 0.0])
        candidates = [
            ("close", np.array([1.1, 0.0])),
            ("far", np.array([5.0, 5.0])),
        ]
        result = translator._mahalanobis_retrieve(
            query, np.eye(2), candidates,
        )
        assert result == "close"

    def test_anisotropic_metric(self, translator):
        """Tensor anisotrópico cambia la distancia."""
        query = np.array([0.0, 0.0])
        G = np.diag([100.0, 1.0])  # Penaliza eje x
        candidates = [
            ("x_close", np.array([0.1, 0.0])),
            ("y_far", np.array([0.0, 1.0])),
        ]
        result = translator._mahalanobis_retrieve(query, G, candidates)
        # d²(x_close) = 0.1² × 100 = 1.0
        # d²(y_far) = 1.0² × 1 = 1.0
        # Ambos iguales, se queda con el primero
        assert result in ["x_close", "y_far"]

    def test_asymmetric_tensor_symmetrized(self, translator):
        """Tensor asimétrico se simetriza automáticamente."""
        query = np.array([1.0, 0.0])
        G = np.array([[1.0, 0.5], [0.0, 1.0]])  # Asimétrico
        candidates = [("a", np.array([0.0, 0.0]))]
        # No debe lanzar error
        result = translator._mahalanobis_retrieve(query, G, candidates)
        assert result == "a"

    def test_dimension_mismatch_fallback(self, translator):
        """Dimensiones incompatibles usa fallback euclidiano."""
        query = np.array([1.0, 0.0])
        G = np.eye(3)  # 3×3 vs query de dim 2
        candidates = [("a", np.array([0.0, 0.0]))]
        result = translator._mahalanobis_retrieve(query, G, candidates)
        assert result == "a"

    def test_candidate_dimension_mismatch_skipped(self, translator):
        """Candidato con dimensión incompatible se salta."""
        query = np.array([1.0, 0.0])
        G = np.eye(2)
        candidates = [
            ("bad", np.array([0.0, 0.0, 0.0])),
            ("good", np.array([0.5, 0.0])),
        ]
        result = translator._mahalanobis_retrieve(query, G, candidates)
        assert result == "good"


class TestSemanticTranslatorHelpers:
    """Verificación de métodos auxiliares."""

    def test_normalize_temperature_finite(self, translator):
        t = translator._normalize_temperature(25.0)
        assert math.isclose(t.celsius, 25.0, abs_tol=0.1)

    def test_normalize_temperature_kelvin_heuristic(self, translator):
        """Valores > 100 se interpretan como Kelvin."""
        t = translator._normalize_temperature(300.0)
        assert t.kelvin == 300.0

    def test_normalize_temperature_nan(self, translator):
        """NaN produce 298.15 K por defecto."""
        t = translator._normalize_temperature(float("nan"))
        assert t.kelvin == 298.15

    def test_safe_extract_numeric(self):
        data = {"a": 42, "b": float("nan"), "c": "string"}
        assert SemanticTranslator._safe_extract_numeric(data, "a") == 42.0
        assert SemanticTranslator._safe_extract_numeric(data, "b") == 0.0
        assert SemanticTranslator._safe_extract_numeric(data, "c") == 0.0
        assert SemanticTranslator._safe_extract_numeric(data, "z") == 0.0

    def test_safe_extract_nested(self):
        data = {"a": {"b": {"c": 3.14}}}
        assert math.isclose(
            SemanticTranslator._safe_extract_nested(data, ["a", "b", "c"]),
            3.14,
        )
        assert (
            SemanticTranslator._safe_extract_nested(data, ["a", "z"])
            == 0.0
        )
        assert (
            SemanticTranslator._safe_extract_nested(data, ["x"])
            == 0.0
        )


# =========================================================================
# GRUPO 12: COMPOSICIÓN ESTRATÉGICA COMPLETA
# =========================================================================
class TestComposeStrategicNarrative:
    """Verificación de la composición del reporte estratégico."""

    def test_compose_viable_project(
        self, translator, valid_topology_connected,
        viable_financial_metrics,
    ):
        """Proyecto completamente viable."""
        report = translator.compose_strategic_narrative(
            topological_metrics=valid_topology_connected,
            financial_metrics=viable_financial_metrics,
            stability=10.0,
        )
        assert isinstance(report, StrategicReport)
        assert report.verdict.is_positive or report.verdict == VerdictLevel.CONDICIONAL
        assert len(report.raw_narrative) > 0
        assert len(report.recommendations) >= 1

    def test_compose_reject_cycles(
        self, translator, valid_topology_with_cycles,
        viable_financial_metrics,
    ):
        """Proyecto con ciclos produce rechazo incondicional."""
        report = translator.compose_strategic_narrative(
            topological_metrics=valid_topology_with_cycles,
            financial_metrics=viable_financial_metrics,
            stability=5.0,
        )
        assert report.verdict == VerdictLevel.RECHAZAR

    def test_compose_reject_instability(
        self, translator, valid_topology_connected,
        viable_financial_metrics,
    ):
        """Ψ < 1.0 produce rechazo incondicional."""
        report = translator.compose_strategic_narrative(
            topological_metrics=valid_topology_connected,
            financial_metrics=viable_financial_metrics,
            stability=0.5,
        )
        assert report.verdict == VerdictLevel.RECHAZAR

    def test_compose_with_all_strata(
        self, translator, valid_topology_connected,
        viable_financial_metrics,
    ):
        """Todos los estratos DIKW están presentes."""
        report = translator.compose_strategic_narrative(
            topological_metrics=valid_topology_connected,
            financial_metrics=viable_financial_metrics,
            stability=10.0,
        )
        assert Stratum.PHYSICS in report.strata_analysis
        assert Stratum.TACTICS in report.strata_analysis
        assert Stratum.STRATEGY in report.strata_analysis
        assert Stratum.OMEGA in report.strata_analysis
        assert Stratum.WISDOM in report.strata_analysis

    def test_compose_with_synergy_risk(
        self, translator, valid_topology_connected,
        viable_financial_metrics,
    ):
        """Sinergia de riesgo produce rechazo."""
        synergy = {
            "synergy_detected": True,
            "intersecting_cycles_count": 2,
        }
        report = translator.compose_strategic_narrative(
            topological_metrics=valid_topology_connected,
            financial_metrics=viable_financial_metrics,
            stability=10.0,
            synergy_risk=synergy,
        )
        assert report.verdict == VerdictLevel.RECHAZAR

    def test_compose_narrative_truncation(self, mock_mic):
        """Narrativa se trunca al máximo configurado."""
        config = TranslatorConfig(max_narrative_length=200)
        t = SemanticTranslator(config=config, mic=mock_mic)
        topo = ValidatedTopology(
            beta_0=1, beta_1=0, beta_2=0,
            euler_characteristic=1,
            fiedler_value=1.0, spectral_gap=0.5,
            pyramid_stability=10.0, structural_entropy=0.1,
        )
        report = t.compose_strategic_narrative(
            topological_metrics=topo,
            financial_metrics={},
            stability=10.0,
        )
        assert len(report.raw_narrative) <= 200

    def test_compose_to_dict_serialization(
        self, translator, valid_topology_connected,
        viable_financial_metrics,
    ):
        """to_dict produce diccionario JSON-serializable."""
        report = translator.compose_strategic_narrative(
            topological_metrics=valid_topology_connected,
            financial_metrics=viable_financial_metrics,
            stability=10.0,
        )
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "verdict" in d
        assert "strata_analysis" in d
        # Debe ser JSON-serializable
        json_str = json.dumps(d, default=str)
        assert len(json_str) > 0

    def test_compose_homomorphism_coherence(
        self, translator, valid_topology_with_cycles,
        viable_financial_metrics,
    ):
        """
        El homomorfismo φ(severity) ≤ final_verdict siempre se cumple.
        """
        report = translator.compose_strategic_narrative(
            topological_metrics=valid_topology_with_cycles,
            financial_metrics=viable_financial_metrics,
            stability=0.5,
        )
        # Con β₁ > 0 y Ψ < 1.0, severity = RECHAZAR
        mapped = SeverityToVerdictHomomorphism.apply(
            SeverityLattice.RECHAZAR
        )
        assert mapped.value <= report.verdict.value

    def test_compose_confidence_range(
        self, translator, valid_topology_connected,
        viable_financial_metrics,
    ):
        """Confianza global ∈ [0, 1]."""
        report = translator.compose_strategic_narrative(
            topological_metrics=valid_topology_connected,
            financial_metrics=viable_financial_metrics,
            stability=10.0,
        )
        assert 0.0 <= report.confidence <= 1.0


class TestComposeStrategicNarrativeLegacy:
    """Verificación de la interfaz legacy."""

    def test_legacy_returns_string(
        self, translator, valid_topology_connected,
        viable_financial_metrics,
    ):
        result = translator.compose_strategic_narrative_legacy(
            topological_metrics=valid_topology_connected,
            financial_metrics=viable_financial_metrics,
            stability=10.0,
        )
        assert isinstance(result, str)
        assert len(result) > 0


class TestAssembleDataProduct:
    """Verificación del ensamblaje de producto de datos."""

    def test_assemble_basic(self, translator, simple_digraph_acyclic):
        report = MagicMock()
        report.strategic_narrative = "Test narrative"
        report.integrity_score = 0.85
        report.complexity_level = "MEDIUM"
        report.waste_alerts = ["alert1"]
        report.circular_risks = []
        report.details = {"key": "value"}

        result = translator.assemble_data_product(
            simple_digraph_acyclic, report,
        )
        assert "metadata" in result
        assert "narrative" in result
        assert "topology" in result
        assert result["topology"]["nodes"] == 4
        assert result["topology"]["edges"] == 4


# =========================================================================
# GRUPO 13: STRATUM ANALYSIS RESULT Y STRATEGIC REPORT
# =========================================================================
class TestStratumAnalysisResult:
    """Verificación de StratumAnalysisResult."""

    def test_valid_construction(self):
        result = StratumAnalysisResult(
            stratum=Stratum.PHYSICS,
            verdict=VerdictLevel.VIABLE,
            narrative="Test",
            metrics_summary={"a": 1},
        )
        assert result.confidence == 1.0
        assert result.severity_score == 0.0

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            StratumAnalysisResult(
                stratum=Stratum.PHYSICS,
                verdict=VerdictLevel.VIABLE,
                narrative="Test",
                metrics_summary={},
                confidence=1.5,
            )

    def test_to_dict(self):
        result = StratumAnalysisResult(
            stratum=Stratum.PHYSICS,
            verdict=VerdictLevel.RECHAZAR,
            narrative="Critical",
            metrics_summary={"temp": 80},
            issues=["overheat"],
        )
        d = result.to_dict()
        assert d["verdict"] == "RECHAZAR"
        assert "overheat" in d["issues"]


class TestStrategicReport:
    """Verificación de StrategicReport."""

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            StrategicReport(
                title="Test",
                verdict=VerdictLevel.VIABLE,
                executive_summary="",
                strata_analysis={},
                recommendations=[],
                raw_narrative="",
                confidence=-0.1,
            )

    def test_is_viable(self):
        report = StrategicReport(
            title="Test",
            verdict=VerdictLevel.VIABLE,
            executive_summary="",
            strata_analysis={},
            recommendations=[],
            raw_narrative="",
        )
        assert report.is_viable

    def test_requires_immediate_action(self):
        report = StrategicReport(
            title="Test",
            verdict=VerdictLevel.RECHAZAR,
            executive_summary="",
            strata_analysis={},
            recommendations=[],
            raw_narrative="",
        )
        assert report.requires_immediate_action


# =========================================================================
# GRUPO 14: FUNCIONES DE UTILIDAD Y FACTORY
# =========================================================================
class TestFactoryAndUtilities:
    """Verificación de funciones de utilidad."""

    def test_create_translator(self, mock_mic):
        t = create_translator(mic=mock_mic)
        assert isinstance(t, SemanticTranslator)

    def test_create_translator_with_config(self, mock_mic):
        config = TranslatorConfig(
            stability=StabilityThresholds(
                critical=2.0, warning=5.0, solid=15.0,
            )
        )
        t = create_translator(config=config, mic=mock_mic)
        assert t.config.stability.critical == 2.0

    def test_verify_verdict_lattice(self):
        assert verify_verdict_lattice()

    def test_verify_severity_homomorphism(self):
        assert verify_severity_homomorphism()

    def test_translate_metrics_to_narrative(self, mock_mic):
        """Función de conveniencia funciona."""
        with patch(
            "app.wisdom.semantic_translator.SemanticTranslator",
        ) as MockTranslator:
            mock_instance = MockTranslator.return_value
            mock_report = MagicMock()
            mock_report.raw_narrative = "test_narrative"
            mock_instance.compose_strategic_narrative.return_value = (
                mock_report
            )
            result = translate_metrics_to_narrative(
                topological_metrics={"beta_0": 1, "beta_1": 0, "beta_2": 0},
                financial_metrics={},
                stability=10.0,
            )
            assert result == "test_narrative"


# =========================================================================
# GRUPO 15: CASOS EXTREMOS Y REGRESIÓN
# =========================================================================
class TestEdgeCases:
    """Casos extremos y regresión."""

    def test_all_exceptions_inherit_from_base(self):
        """Todas las excepciones del dominio heredan de la base."""
        assert issubclass(
            TopologyInvariantViolation, SemanticTranslatorError
        )
        assert issubclass(LatticeViolation, SemanticTranslatorError)
        assert issubclass(MetricsValidationError, SemanticTranslatorError)
        assert issubclass(GraphStructureError, SemanticTranslatorError)

    def test_has_betti_numbers_protocol(self, valid_topology_connected):
        """ValidatedTopology satisface el protocolo HasBettiNumbers."""
        assert isinstance(valid_topology_connected, HasBettiNumbers)

    def test_verdict_monotonicity_under_join(self):
        """Propiedad: a ≤ a ⊔ b ∀a,b."""
        for a in VerdictLevel:
            for b in VerdictLevel:
                assert a <= (a | b)
                assert b <= (a | b)

    def test_verdict_monotonicity_under_meet(self):
        """Propiedad: a ⊓ b ≤ a ∀a,b."""
        for a in VerdictLevel:
            for b in VerdictLevel:
                assert (a & b) <= a
                assert (a & b) <= b

    def test_temperature_conversion_roundtrip(self):
        """Celsius → Kelvin → Celsius es idempotente."""
        original_celsius = 37.5
        t = Temperature.from_celsius(original_celsius)
        roundtrip = t.celsius
        assert math.isclose(roundtrip, original_celsius, abs_tol=1e-10)

    def test_fiedler_hysteresis(self, translator, valid_topology_connected):
        """Histéresis de Fiedler previene oscilaciones."""
        # Primera evaluación establece baseline
        topo1 = ValidatedTopology(
            beta_0=1, beta_1=0, beta_2=0,
            euler_characteristic=1,
            fiedler_value=0.5,
            spectral_gap=0.3,
            pyramid_stability=10.0,
            structural_entropy=0.2,
        )
        translator.translate_topology(topo1, stability=10.0)

        # Segunda evaluación con cambio minúsculo dentro de banda
        topo2 = ValidatedTopology(
            beta_0=1, beta_1=0, beta_2=0,
            euler_characteristic=1,
            fiedler_value=0.5 + 1e-13,  # Dentro de banda de histéresis
            spectral_gap=0.3,
            pyramid_stability=10.0,
            structural_entropy=0.2,
        )
        translator.translate_topology(topo2, stability=10.0)
        # Debe usar max(current, last) por histéresis
        assert translator._last_fiedler >= 0.5

    def test_dikw_filtration_ordering(
        self, translator, valid_topology_with_cycles,
        viable_financial_metrics,
    ):
        """
        Filtración DIKW: si PHYSICS/TACTICS falla, WISDOM hereda el fallo.

        V_PHYSICS ⊂ V_TACTICS ⊂ V_STRATEGY ⊂ V_WISDOM
        Un fallo en la base corrompe el estrato superior.
        """
        report = translator.compose_strategic_narrative(
            topological_metrics=valid_topology_with_cycles,
            financial_metrics=viable_financial_metrics,
            stability=0.5,
        )
        tactics_verdict = report.strata_analysis[Stratum.TACTICS].verdict
        wisdom_verdict = report.strata_analysis[Stratum.WISDOM].verdict

        # WISDOM debe ser al menos tan severo como TACTICS
        assert wisdom_verdict.value >= tactics_verdict.value

    def test_market_context_with_provider(self, mock_mic):
        """Market provider inyectado se usa correctamente."""
        provider = MagicMock(return_value="Mercado estable")
        t = SemanticTranslator(
            mic=mock_mic, market_provider=provider,
        )
        result = t.get_market_context()
        assert "Mercado estable" in result
        provider.assert_called_once()

    def test_market_context_provider_failure(self, mock_mic):
        """Fallo del market provider produce fallback."""
        provider = MagicMock(side_effect=RuntimeError("API down"))
        t = SemanticTranslator(
            mic=mock_mic, market_provider=provider,
        )
        result = t.get_market_context()
        assert "No disponible" in result

    def test_explain_cycle_path_empty(self, translator):
        assert translator.explain_cycle_path([]) == ""

    def test_explain_cycle_path_truncation(self, translator):
        """Ciclo largo se trunca correctamente."""
        long_cycle = [f"N{i}" for i in range(20)]
        # No debe fallar, aunque la narrativa depende del MIC
        result = translator.explain_cycle_path(long_cycle)
        assert isinstance(result, str)

    def test_explain_stress_point_numeric(self, translator):
        result = translator.explain_stress_point("NODE_X", 15)
        assert isinstance(result, str)

    def test_explain_stress_point_string_degree(self, translator):
        result = translator.explain_stress_point("NODE_Y", "5")
        assert isinstance(result, str)

    def test_explain_stress_point_qualitative(self, translator):
        """Grado cualitativo usa valor alto por defecto."""
        result = translator.explain_stress_point(
            "NODE_Z", "múltiples",
        )
        assert isinstance(result, str)

    def test_no_cache_mode(self, translator_no_cache):
        """Traductor sin caché funciona correctamente."""
        assert translator_no_cache._cache is None
        # Debe funcionar sin error
        topo = ValidatedTopology(
            beta_0=1, beta_1=0, beta_2=0,
            euler_characteristic=1,
            fiedler_value=1.0, spectral_gap=0.5,
            pyramid_stability=10.0, structural_entropy=0.1,
        )
        narrative, verdict = translator_no_cache.translate_topology(
            topo, stability=10.0,
        )
        assert len(narrative) > 0

    def test_severity_score_continuity(self):
        """severity_score es continua y monótona en [0, solid]."""
        st = StabilityThresholds()
        prev_score = st.severity_score(0.0)
        for i in range(1, 100):
            stability = i * st.solid / 100.0
            score = st.severity_score(stability)
            assert score <= prev_score, (
                f"Monotonicity violated at Ψ={stability}"
            )
            prev_score = score

    def test_large_graph_degree_fallback(self):
        """Grafos grandes usan fallback de grado."""
        G = nx.DiGraph()
        # Crear grafo grande (> MAX_GRAPH_NODES_FOR_EIGENVECTOR sería
        # demasiado, así que verificamos el path del fallback)
        G.add_edges_from([(str(i), str(i + 1)) for i in range(10)])
        narrator = GraphRAGCausalNarrator(G)
        node = narrator._find_critical_node_by_degree()
        assert node in [str(i) for i in range(11)]

    def test_validate_financial_metrics_robust(self, translator):
        """Métricas financieras con campos faltantes no fallan."""
        metrics = {"wacc": 0.05}  # Muchos campos faltantes
        result = translator._validate_financial_metrics(metrics)
        assert "wacc" in result
        assert result["wacc"] == 0.05
        assert result["contingency_recommended"] == 0.0


# =========================================================================
# GRUPO 16: PROPIEDADES ALGEBRAICAS DE INTEGRACIÓN
# =========================================================================
class TestAlgebraicIntegrationProperties:
    """
    Propiedades algebraicas de integración entre componentes.

    Verifica que las composiciones funcionales preservan las
    propiedades del retículo y los homomorfismos.
    """

    def test_supremum_idempotent_full_pipeline(
        self, translator, valid_topology_connected,
        viable_financial_metrics,
    ):
        """
        Propiedad: ejecutar compose dos veces con mismos datos
        produce el mismo veredicto (determinismo).
        """
        report1 = translator.compose_strategic_narrative(
            topological_metrics=valid_topology_connected,
            financial_metrics=viable_financial_metrics,
            stability=10.0,
        )
        # Reset histéresis para determinismo
        translator._last_fiedler = None
        translator._last_verdict = None

        report2 = translator.compose_strategic_narrative(
            topological_metrics=valid_topology_connected,
            financial_metrics=viable_financial_metrics,
            stability=10.0,
        )
        assert report1.verdict == report2.verdict

    def test_lattice_collapse_absorbing_in_pipeline(
        self, translator, valid_topology_with_cycles,
        viable_financial_metrics,
    ):
        """
        Propiedad absorbente: si cualquier estrato emite ⊤ (RECHAZAR),
        el veredicto final es ⊤ independientemente de los demás.
        """
        report = translator.compose_strategic_narrative(
            topological_metrics=valid_topology_with_cycles,
            financial_metrics=viable_financial_metrics,
            stability=10.0,
        )
        # β₁ = 2 > 0 ⟹ veto incondicional ⟹ RECHAZAR
        assert report.verdict == VerdictLevel.RECHAZAR
        assert report.verdict == VerdictLevel.top()

    def test_homomorphism_composition_coherent(self):
        """
        Composición: FinancialVerdict → VerdictLevel preserva orden.

        Si fv1 es "mejor" que fv2 en FinancialVerdict,
        entonces to_verdict_level(fv1) ≤ to_verdict_level(fv2).
        """
        fv_order = [
            FinancialVerdict.ACCEPT,
            FinancialVerdict.CONDITIONAL,
            FinancialVerdict.REVIEW,
            FinancialVerdict.REJECT,
        ]
        for i in range(len(fv_order)):
            for j in range(i, len(fv_order)):
                vl_i = fv_order[i].to_verdict_level()
                vl_j = fv_order[j].to_verdict_level()
                assert vl_i <= vl_j, (
                    f"Order not preserved: "
                    f"{fv_order[i]} → {vl_i} vs "
                    f"{fv_order[j]} → {vl_j}"
                )

    def test_severity_lattice_laws(self):
        """
        SeverityLattice satisface leyes de retículo (3 elementos).
        """
        elems = list(SeverityLattice)
        # Idempotencia
        for a in elems:
            assert SeverityLattice.supremum(a, a) == a
            assert SeverityLattice.infimum(a, a) == a
        # Conmutatividad
        for a in elems:
            for b in elems:
                assert (
                    SeverityLattice.supremum(a, b)
                    == SeverityLattice.supremum(b, a)
                )
                assert (
                    SeverityLattice.infimum(a, b)
                    == SeverityLattice.infimum(b, a)
                )
        # Absorción: a ⊔ (a ⊓ b) = a
        for a in elems:
            for b in elems:
                meet_ab = SeverityLattice.infimum(a, b)
                assert SeverityLattice.supremum(a, meet_ab) == a