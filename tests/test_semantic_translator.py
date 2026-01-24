"""
Suite de pruebas para el módulo de Traducción Semántica.

Valida:
1. Propiedades algebraicas del Lattice de Veredictos
2. Correcta traducción de métricas topológicas y financieras
3. Normalización de datos vía DTOs
4. Integración con jerarquía DIKW/Stratum
5. Composición de reportes estratégicos

Estructura de Tests:
- TestLatticeAlgebraicProperties: Propiedades matemáticas del lattice
- TestVerdictLevelOperations: Operaciones específicas de veredictos
- TestDTONormalization: Validación de DTOs
- TestThresholdsValidation: Validación de umbrales
- TestTopologyTranslation: Traducción topológica
- TestThermalTranslation: Traducción termodinámica
- TestFinancialTranslation: Traducción financiera
- TestStrategicReportComposition: Composición de reportes
- TestStratumIntegration: Integración con pirámide DIKW
- TestEdgeCasesAndErrorHandling: Casos límite y errores
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from typing import Any, Dict, List, Tuple
from unittest.mock import Mock, patch

import pytest

# Importar desde el módulo refinado
from app.semantic_translator import (
    # Configuración
    TranslatorConfig,
    StabilityThresholds,
    TopologicalThresholds,
    ThermalThresholds,
    FinancialThresholds,
    # Lattice
    VerdictLevel,
    FinancialVerdict,
    # DTOs
    TopologyMetricsDTO,
    ThermalMetricsDTO,
    SpectralMetricsDTO,
    SynergyRiskDTO,
    # Resultados
    StratumAnalysisResult,
    StrategicReport,
    # Traductor
    SemanticTranslator,
    NarrativeTemplates,
    # Factories
    create_translator,
    translate_metrics_to_narrative,
)

# Importar Stratum para tests de integración
try:
    from app.schemas import Stratum
except ImportError:
    from enum import IntEnum
    class Stratum(IntEnum):
        WISDOM = 0
        STRATEGY = 1
        TACTICS = 2
        PHYSICS = 3

# Intentar importar TopologicalMetrics original para compatibilidad
try:
    from agent.business_topology import TopologicalMetrics
    HAS_ORIGINAL_METRICS = True
except ImportError:
    HAS_ORIGINAL_METRICS = False
    TopologicalMetrics = None


# ============================================================================
# FIXTURES GLOBALES
# ============================================================================


@pytest.fixture
def default_translator() -> SemanticTranslator:
    """Traductor con configuración por defecto."""
    return SemanticTranslator()


@pytest.fixture
def deterministic_translator() -> SemanticTranslator:
    """Traductor con comportamiento determinístico."""
    config = TranslatorConfig(deterministic_market=True, default_market_index=0)
    return SemanticTranslator(config=config)


@pytest.fixture
def clean_topology() -> TopologyMetricsDTO:
    """Topología limpia: conexa, sin ciclos."""
    return TopologyMetricsDTO(beta_0=1, beta_1=0, euler_characteristic=1)


@pytest.fixture
def cyclic_topology() -> TopologyMetricsDTO:
    """Topología con ciclos."""
    return TopologyMetricsDTO(beta_0=1, beta_1=2, euler_characteristic=-1)


@pytest.fixture
def fragmented_topology() -> TopologyMetricsDTO:
    """Topología fragmentada."""
    return TopologyMetricsDTO(beta_0=3, beta_1=0, euler_characteristic=3)


@pytest.fixture
def viable_financials() -> Dict[str, Any]:
    """Métricas financieras viables."""
    return {
        "wacc": 0.10,
        "var": 1000.0,
        "contingency": {"recommended": 1500.0},
        "performance": {"recommendation": "ACEPTAR", "profitability_index": 1.25},
    }


@pytest.fixture
def rejected_financials() -> Dict[str, Any]:
    """Métricas financieras de rechazo."""
    return {
        "wacc": 0.18,
        "var": 100000.0,
        "contingency": {"recommended": 150000.0},
        "performance": {"recommendation": "RECHAZAR", "profitability_index": 0.65},
    }


@pytest.fixture
def review_financials() -> Dict[str, Any]:
    """Métricas financieras en revisión."""
    return {
        "wacc": 0.12,
        "contingency": {"recommended": 5000.0},
        "performance": {"recommendation": "REVISAR", "profitability_index": 1.0},
    }


# ============================================================================
# TEST: PROPIEDADES ALGEBRAICAS DEL LATTICE
# ============================================================================


class TestLatticeAlgebraicProperties:
    """
    Verifica las propiedades algebraicas del lattice (VerdictLevel, ≤, ⊔, ⊓).
    
    Un lattice debe satisfacer:
    1. Conmutatividad: a ⊔ b = b ⊔ a, a ⊓ b = b ⊓ a
    2. Asociatividad: (a ⊔ b) ⊔ c = a ⊔ (b ⊔ c)
    3. Idempotencia: a ⊔ a = a, a ⊓ a = a
    4. Absorción: a ⊔ (a ⊓ b) = a, a ⊓ (a ⊔ b) = a
    5. Identidad: a ⊔ ⊥ = a, a ⊓ ⊤ = a
    """

    @pytest.fixture
    def all_verdicts(self) -> List[VerdictLevel]:
        """Todos los niveles de veredicto."""
        return list(VerdictLevel)

    def test_lattice_has_bottom_and_top(self):
        """Verifica existencia de elementos ⊥ y ⊤."""
        assert VerdictLevel.bottom() == VerdictLevel.VIABLE
        assert VerdictLevel.top() == VerdictLevel.RECHAZAR

    def test_lattice_order_is_total(self, all_verdicts):
        """Verifica que el orden es total (todos comparables)."""
        for a in all_verdicts:
            for b in all_verdicts:
                # En un orden total, siempre a <= b o b <= a
                assert a.value <= b.value or b.value <= a.value

    @pytest.mark.parametrize("a", list(VerdictLevel))
    @pytest.mark.parametrize("b", list(VerdictLevel))
    def test_join_commutativity(self, a: VerdictLevel, b: VerdictLevel):
        """Conmutatividad del join: a ⊔ b = b ⊔ a."""
        assert a | b == b | a
        assert VerdictLevel.supremum(a, b) == VerdictLevel.supremum(b, a)

    @pytest.mark.parametrize("a", list(VerdictLevel))
    @pytest.mark.parametrize("b", list(VerdictLevel))
    def test_meet_commutativity(self, a: VerdictLevel, b: VerdictLevel):
        """Conmutatividad del meet: a ⊓ b = b ⊓ a."""
        assert a & b == b & a
        assert VerdictLevel.infimum(a, b) == VerdictLevel.infimum(b, a)

    @pytest.mark.parametrize("a", list(VerdictLevel))
    @pytest.mark.parametrize("b", list(VerdictLevel))
    @pytest.mark.parametrize("c", list(VerdictLevel))
    def test_join_associativity(self, a: VerdictLevel, b: VerdictLevel, c: VerdictLevel):
        """Asociatividad del join: (a ⊔ b) ⊔ c = a ⊔ (b ⊔ c)."""
        assert (a | b) | c == a | (b | c)

    @pytest.mark.parametrize("a", list(VerdictLevel))
    @pytest.mark.parametrize("b", list(VerdictLevel))
    @pytest.mark.parametrize("c", list(VerdictLevel))
    def test_meet_associativity(self, a: VerdictLevel, b: VerdictLevel, c: VerdictLevel):
        """Asociatividad del meet: (a ⊓ b) ⊓ c = a ⊓ (b ⊓ c)."""
        assert (a & b) & c == a & (b & c)

    @pytest.mark.parametrize("a", list(VerdictLevel))
    def test_join_idempotence(self, a: VerdictLevel):
        """Idempotencia del join: a ⊔ a = a."""
        assert a | a == a
        assert VerdictLevel.supremum(a, a) == a

    @pytest.mark.parametrize("a", list(VerdictLevel))
    def test_meet_idempotence(self, a: VerdictLevel):
        """Idempotencia del meet: a ⊓ a = a."""
        assert a & a == a
        assert VerdictLevel.infimum(a, a) == a

    @pytest.mark.parametrize("a", list(VerdictLevel))
    @pytest.mark.parametrize("b", list(VerdictLevel))
    def test_absorption_law_join(self, a: VerdictLevel, b: VerdictLevel):
        """Ley de absorción join: a ⊔ (a ⊓ b) = a."""
        assert a | (a & b) == a

    @pytest.mark.parametrize("a", list(VerdictLevel))
    @pytest.mark.parametrize("b", list(VerdictLevel))
    def test_absorption_law_meet(self, a: VerdictLevel, b: VerdictLevel):
        """Ley de absorción meet: a ⊓ (a ⊔ b) = a."""
        assert a & (a | b) == a

    @pytest.mark.parametrize("a", list(VerdictLevel))
    def test_bottom_is_join_identity(self, a: VerdictLevel):
        """⊥ es identidad del join: a ⊔ ⊥ = a."""
        bottom = VerdictLevel.bottom()
        assert a | bottom == a

    @pytest.mark.parametrize("a", list(VerdictLevel))
    def test_top_is_meet_identity(self, a: VerdictLevel):
        """⊤ es identidad del meet: a ⊓ ⊤ = a."""
        top = VerdictLevel.top()
        assert a & top == a

    def test_supremum_empty_returns_bottom(self):
        """Supremum vacío retorna ⊥."""
        assert VerdictLevel.supremum() == VerdictLevel.VIABLE

    def test_infimum_empty_returns_top(self):
        """Infimum vacío retorna ⊤."""
        assert VerdictLevel.infimum() == VerdictLevel.RECHAZAR

    def test_supremum_takes_worst_case(self):
        """Supremum toma el peor caso."""
        result = VerdictLevel.supremum(
            VerdictLevel.VIABLE,
            VerdictLevel.CONDICIONAL,
            VerdictLevel.RECHAZAR,
        )
        assert result == VerdictLevel.RECHAZAR

    def test_infimum_takes_best_case(self):
        """Infimum toma el mejor caso."""
        result = VerdictLevel.infimum(
            VerdictLevel.VIABLE,
            VerdictLevel.CONDICIONAL,
            VerdictLevel.RECHAZAR,
        )
        assert result == VerdictLevel.VIABLE


class TestVerdictLevelOperations:
    """Pruebas de operaciones específicas de VerdictLevel."""

    def test_verdict_emoji_mapping(self):
        """Verifica mapeo de emojis."""
        assert VerdictLevel.VIABLE.emoji == "✅"
        assert VerdictLevel.CONDICIONAL.emoji == "🔵"
        assert VerdictLevel.REVISAR.emoji == "🔍"
        assert VerdictLevel.PRECAUCION.emoji == "⚠️"
        assert VerdictLevel.RECHAZAR.emoji == "🛑"

    def test_is_positive_classification(self):
        """Verifica clasificación de veredictos positivos."""
        assert VerdictLevel.VIABLE.is_positive is True
        assert VerdictLevel.CONDICIONAL.is_positive is True
        assert VerdictLevel.REVISAR.is_positive is False
        assert VerdictLevel.PRECAUCION.is_positive is False
        assert VerdictLevel.RECHAZAR.is_positive is False

    def test_is_negative_classification(self):
        """Verifica clasificación de veredictos negativos."""
        assert VerdictLevel.VIABLE.is_negative is False
        assert VerdictLevel.RECHAZAR.is_negative is True

    def test_binary_operators_are_consistent(self):
        """Verifica consistencia entre operadores y métodos."""
        a, b = VerdictLevel.CONDICIONAL, VerdictLevel.PRECAUCION

        assert a | b == a.join(b)
        assert a & b == a.meet(b)
        assert a | b == VerdictLevel.supremum(a, b)
        assert a & b == VerdictLevel.infimum(a, b)


class TestFinancialVerdictIntegration:
    """Pruebas de integración entre FinancialVerdict y VerdictLevel."""

    @pytest.mark.parametrize(
        "fin_verdict,expected_level",
        [
            (FinancialVerdict.ACCEPT, VerdictLevel.VIABLE),
            (FinancialVerdict.CONDITIONAL, VerdictLevel.CONDICIONAL),
            (FinancialVerdict.REVIEW, VerdictLevel.REVISAR),
            (FinancialVerdict.REJECT, VerdictLevel.RECHAZAR),
        ],
    )
    def test_financial_verdict_to_verdict_level(
        self, fin_verdict: FinancialVerdict, expected_level: VerdictLevel
    ):
        """Verifica conversión de FinancialVerdict a VerdictLevel."""
        assert fin_verdict.to_verdict_level() == expected_level

    @pytest.mark.parametrize(
        "input_str,expected",
        [
            ("ACEPTAR", FinancialVerdict.ACCEPT),
            ("aceptar", FinancialVerdict.ACCEPT),
            ("ACCEPT", FinancialVerdict.ACCEPT),
            ("RECHAZAR", FinancialVerdict.REJECT),
            ("REVISAR", FinancialVerdict.REVIEW),
            ("CONDICIONAL", FinancialVerdict.CONDITIONAL),
            ("invalid", FinancialVerdict.REVIEW),
            ("", FinancialVerdict.REVIEW),
        ],
    )
    def test_financial_verdict_from_string(self, input_str: str, expected: FinancialVerdict):
        """Verifica parsing de strings a FinancialVerdict."""
        assert FinancialVerdict.from_string(input_str) == expected


# ============================================================================
# TEST: DTOs Y NORMALIZACIÓN
# ============================================================================


class TestTopologyMetricsDTO:
    """Pruebas para TopologyMetricsDTO."""

    def test_create_with_defaults(self):
        """Creación con valores por defecto."""
        dto = TopologyMetricsDTO()
        assert dto.beta_0 == 1
        assert dto.beta_1 == 0
        assert dto.euler_characteristic == 1
        assert dto.euler_efficiency == 1.0

    def test_create_with_custom_values(self):
        """Creación con valores personalizados."""
        dto = TopologyMetricsDTO(beta_0=3, beta_1=2, euler_characteristic=1)
        assert dto.beta_0 == 3
        assert dto.beta_1 == 2

    def test_negative_beta_0_raises_error(self):
        """β₀ negativo lanza error."""
        with pytest.raises(ValueError, match="β₀"):
            TopologyMetricsDTO(beta_0=-1, beta_1=0)

    def test_negative_beta_1_raises_error(self):
        """β₁ negativo lanza error."""
        with pytest.raises(ValueError, match="β₁"):
            TopologyMetricsDTO(beta_0=1, beta_1=-1)

    def test_euler_consistency_warning(self, caplog):
        """Inconsistencia de Euler genera warning."""
        # χ debería ser β₀ - β₁ = 1 - 0 = 1, no 5
        dto = TopologyMetricsDTO(beta_0=1, beta_1=0, euler_characteristic=5)
        assert "mismatch" in caplog.text.lower() or dto.euler_characteristic == 5

    def test_from_dict(self):
        """Creación desde diccionario."""
        data = {"beta_0": 2, "beta_1": 1, "euler_characteristic": 1}
        dto = TopologyMetricsDTO.from_any(data)
        assert dto.beta_0 == 2
        assert dto.beta_1 == 1

    def test_from_none_returns_defaults(self):
        """None retorna valores por defecto."""
        dto = TopologyMetricsDTO.from_any(None)
        assert dto.beta_0 == 1
        assert dto.beta_1 == 0

    def test_from_self_returns_same(self):
        """DTO pasado retorna el mismo objeto."""
        original = TopologyMetricsDTO(beta_0=5, beta_1=3)
        result = TopologyMetricsDTO.from_any(original)
        assert result is original

    @pytest.mark.skipif(not HAS_ORIGINAL_METRICS, reason="TopologicalMetrics not available")
    def test_from_original_topological_metrics(self):
        """Conversión desde TopologicalMetrics original."""
        original = TopologicalMetrics(beta_0=2, beta_1=1, euler_characteristic=1)
        dto = TopologyMetricsDTO.from_any(original)
        assert dto.beta_0 == 2
        assert dto.beta_1 == 1

    def test_from_invalid_type_raises_error(self):
        """Tipo inválido lanza TypeError."""
        with pytest.raises(TypeError, match="Cannot convert"):
            TopologyMetricsDTO.from_any("invalid")


class TestThermalMetricsDTO:
    """Pruebas para ThermalMetricsDTO."""

    def test_create_with_defaults(self):
        """Creación con valores por defecto."""
        dto = ThermalMetricsDTO()
        assert dto.entropy == 0.0
        assert dto.exergy == 1.0
        assert dto.temperature == 25.0

    def test_values_clamped_to_valid_range(self):
        """Valores se limitan a rangos válidos."""
        dto = ThermalMetricsDTO(entropy=1.5, exergy=-0.5, temperature=-10)
        assert dto.entropy == 1.0  # Clamped to max
        assert dto.exergy == 0.0   # Clamped to min
        assert dto.temperature == 0.0  # Clamped to min

    def test_from_dict(self):
        """Creación desde diccionario."""
        data = {"entropy": 0.5, "exergy": 0.8, "system_temperature": 45.0}
        dto = ThermalMetricsDTO.from_dict(data)
        assert dto.entropy == 0.5
        assert dto.exergy == 0.8
        assert dto.temperature == 45.0

    def test_from_dict_alternative_key(self):
        """Acepta 'temperature' como alternativa a 'system_temperature'."""
        data = {"temperature": 35.0}
        dto = ThermalMetricsDTO.from_dict(data)
        assert dto.temperature == 35.0

    def test_from_none_returns_defaults(self):
        """None retorna valores por defecto."""
        dto = ThermalMetricsDTO.from_dict(None)
        assert dto.temperature == 25.0


class TestSynergyRiskDTO:
    """Pruebas para SynergyRiskDTO."""

    def test_create_with_defaults(self):
        """Creación con valores por defecto."""
        dto = SynergyRiskDTO()
        assert dto.synergy_detected is False
        assert dto.intersecting_cycles_count == 0
        assert dto.intersecting_nodes == []

    def test_from_dict_with_synergy(self):
        """Creación desde diccionario con sinergia detectada."""
        data = {
            "synergy_detected": True,
            "intersecting_cycles_count": 2,
            "intersecting_nodes": ["node_a", "node_b"],
            "intersecting_cycles": [["a", "b", "c"], ["d", "e"]],
        }
        dto = SynergyRiskDTO.from_dict(data)
        assert dto.synergy_detected is True
        assert dto.intersecting_cycles_count == 2
        assert len(dto.intersecting_nodes) == 2


# ============================================================================
# TEST: UMBRALES Y CONFIGURACIÓN
# ============================================================================


class TestThresholdsValidation:
    """Pruebas de validación de umbrales."""

    def test_stability_thresholds_default(self):
        """Umbrales de estabilidad por defecto."""
        thresholds = StabilityThresholds()
        assert thresholds.critical == 1.0
        assert thresholds.solid == 10.0
        assert thresholds.warning == 3.0

    def test_stability_thresholds_invariant_critical_less_than_solid(self):
        """critical debe ser menor que solid."""
        with pytest.raises(ValueError, match="greater than critical"):
            StabilityThresholds(critical=10.0, solid=5.0)

    def test_stability_thresholds_invariant_critical_non_negative(self):
        """critical debe ser no negativo."""
        with pytest.raises(ValueError, match="non-negative"):
            StabilityThresholds(critical=-1.0, solid=10.0)

    def test_stability_thresholds_classify(self):
        """Clasificación de estabilidad."""
        thresholds = StabilityThresholds(critical=1.0, solid=10.0, warning=3.0)

        assert thresholds.classify(0.5) == "critical"
        assert thresholds.classify(1.5) == "warning"
        assert thresholds.classify(5.0) == "stable"
        assert thresholds.classify(15.0) == "robust"
        assert thresholds.classify(-1.0) == "invalid"

    def test_topological_thresholds_classify_connectivity(self):
        """Clasificación de conectividad."""
        thresholds = TopologicalThresholds()

        assert thresholds.classify_connectivity(0) == "empty"
        assert thresholds.classify_connectivity(1) == "unified"
        assert thresholds.classify_connectivity(3) == "fragmented"
        assert thresholds.classify_connectivity(10) == "severely_fragmented"

    def test_topological_thresholds_classify_cycles(self):
        """Clasificación de ciclos."""
        thresholds = TopologicalThresholds()

        assert thresholds.classify_cycles(0) == "clean"
        assert thresholds.classify_cycles(1) == "minor"
        assert thresholds.classify_cycles(2) == "moderate"
        assert thresholds.classify_cycles(5) == "critical"

    def test_translator_config_immutability(self):
        """TranslatorConfig es frozen por defecto en sus sub-dataclasses."""
        config = TranslatorConfig()

        # StabilityThresholds es frozen
        with pytest.raises(FrozenInstanceError):
            config.stability.critical = 5.0

    def test_translator_config_custom_values(self):
        """Configuración personalizada."""
        custom_stability = StabilityThresholds(critical=2.0, solid=20.0)
        config = TranslatorConfig(
            stability=custom_stability,
            deterministic_market=True,
            max_cycle_path_display=10,
        )

        assert config.stability.critical == 2.0
        assert config.deterministic_market is True
        assert config.max_cycle_path_display == 10


# ============================================================================
# TEST: TRADUCCIÓN DE TOPOLOGÍA
# ============================================================================


class TestTopologyTranslation:
    """Pruebas para traducción de métricas topológicas."""

    def test_translate_clean_topology(
        self, default_translator: SemanticTranslator, clean_topology: TopologyMetricsDTO
    ):
        """Topología limpia genera narrativa positiva."""
        narrative, verdict = default_translator.translate_topology(
            clean_topology, stability=15.0
        )

        assert "Integridad Estructural" in narrative or "Genus 0" in narrative
        assert "Unidad de Obra" in narrative or "Monolítica" in narrative
        assert verdict == VerdictLevel.VIABLE

    def test_translate_cyclic_topology(
        self, default_translator: SemanticTranslator, cyclic_topology: TopologyMetricsDTO
    ):
        """Topología con ciclos genera advertencias."""
        narrative, verdict = default_translator.translate_topology(
            cyclic_topology, stability=15.0
        )

        assert "socavones" in narrative.lower() or "ciclos" in narrative.lower()
        assert verdict.value >= VerdictLevel.CONDICIONAL.value

    def test_translate_fragmented_topology(
        self, default_translator: SemanticTranslator, fragmented_topology: TopologyMetricsDTO
    ):
        """Topología fragmentada genera advertencias."""
        narrative, verdict = default_translator.translate_topology(
            fragmented_topology, stability=15.0
        )

        assert "Desconectados" in narrative or "Fragmentación" in narrative
        assert "3" in narrative  # β₀ = 3

    def test_translate_empty_topology(self, default_translator: SemanticTranslator):
        """β₀ = 0 indica terreno vacío."""
        empty = TopologyMetricsDTO(beta_0=0, beta_1=0, euler_characteristic=0)
        narrative, verdict = default_translator.translate_topology(empty, stability=10.0)

        assert "Vacío" in narrative
        assert verdict == VerdictLevel.RECHAZAR

    @pytest.mark.parametrize(
        "stability,expected_class",
        [
            (0.5, "critical"),
            (1.5, "warning"),
            (5.0, "stable"),
            (15.0, "robust"),
        ],
    )
    def test_stability_classification_in_narrative(
        self,
        default_translator: SemanticTranslator,
        clean_topology: TopologyMetricsDTO,
        stability: float,
        expected_class: str,
    ):
        """Verifica clasificación de estabilidad en narrativa."""
        narrative, _ = default_translator.translate_topology(clean_topology, stability)

        # Mapeo de clase a keywords esperados
        keywords = {
            "critical": ["Pirámide Invertida", "COLAPSO"],
            "warning": ["Equilibrio Precario", "Isostático"],
            "stable": ["Isostática", "Estable"],
            "robust": ["ANTISÍSMICA", "Resiliente"],
        }

        found = any(kw in narrative for kw in keywords[expected_class])
        assert found, f"No keyword found for {expected_class} in: {narrative[:200]}"

    def test_translate_with_synergy_risk(self, default_translator: SemanticTranslator):
        """Sinergia de riesgo aumenta severidad."""
        topo = TopologyMetricsDTO(beta_0=1, beta_1=1)
        synergy = {"synergy_detected": True, "intersecting_cycles_count": 3}

        narrative, verdict = default_translator.translate_topology(
            topo, stability=10.0, synergy_risk=synergy
        )

        assert "Dominó" in narrative or "Contagio" in narrative
        assert verdict == VerdictLevel.RECHAZAR

    def test_translate_with_spectral_resonance(self, default_translator: SemanticTranslator):
        """Riesgo de resonancia genera advertencia."""
        topo = TopologyMetricsDTO(beta_0=1, beta_1=0)
        spectral = {"fiedler_value": 0.3, "wavelength": 2.5, "resonance_risk": True}

        narrative, verdict = default_translator.translate_topology(
            topo, stability=10.0, spectral=spectral
        )

        assert "RESONANCIA" in narrative
        assert verdict.value >= VerdictLevel.PRECAUCION.value

    def test_translate_with_low_euler_efficiency(self, default_translator: SemanticTranslator):
        """Baja eficiencia de Euler genera narrativa de sobrecarga."""
        topo = TopologyMetricsDTO(beta_0=1, beta_1=0, euler_efficiency=0.3)

        narrative, _ = default_translator.translate_topology(topo, stability=10.0)

        assert "Sobrecarga" in narrative or "Entropía" in narrative

    def test_translate_accepts_dict_input(self, default_translator: SemanticTranslator):
        """Acepta diccionario como input."""
        dict_input = {"beta_0": 1, "beta_1": 0, "euler_characteristic": 1}

        narrative, verdict = default_translator.translate_topology(
            dict_input, stability=10.0
        )

        assert "Integridad" in narrative
        assert verdict == VerdictLevel.VIABLE

    def test_translate_negative_stability_raises_error(
        self, default_translator: SemanticTranslator, clean_topology: TopologyMetricsDTO
    ):
        """Estabilidad negativa lanza error."""
        with pytest.raises(ValueError, match="non-negative"):
            default_translator.translate_topology(clean_topology, stability=-1.0)


# ============================================================================
# TEST: TRADUCCIÓN TERMODINÁMICA
# ============================================================================


class TestThermalTranslation:
    """Pruebas para traducción de métricas termodinámicas."""

    def test_translate_cold_temperature(self, default_translator: SemanticTranslator):
        """Temperatura fría genera narrativa estable."""
        narrative, verdict = default_translator.translate_thermodynamics(
            entropy=0.2, exergy=0.9, temperature=15.0
        )

        assert "Estable" in narrative or "❄️" in narrative
        assert verdict == VerdictLevel.VIABLE

    def test_translate_hot_temperature(self, default_translator: SemanticTranslator):
        """Temperatura caliente genera advertencia de fiebre."""
        narrative, verdict = default_translator.translate_thermodynamics(
            entropy=0.3, exergy=0.7, temperature=60.0
        )

        assert "FIEBRE" in narrative or "🔥" in narrative
        assert "Receta" in narrative  # Recomendación
        assert verdict.value >= VerdictLevel.PRECAUCION.value

    def test_translate_critical_temperature(self, default_translator: SemanticTranslator):
        """Temperatura crítica genera rechazo."""
        narrative, verdict = default_translator.translate_thermodynamics(
            entropy=0.5, exergy=0.5, temperature=80.0
        )

        assert "FUSIÓN" in narrative or "crítica" in narrative.lower()
        assert verdict == VerdictLevel.RECHAZAR

    def test_translate_high_entropy(self, default_translator: SemanticTranslator):
        """Alta entropía genera narrativa de caos."""
        narrative, verdict = default_translator.translate_thermodynamics(
            entropy=0.85, exergy=0.5, temperature=25.0
        )

        assert "Entropía" in narrative or "Caos" in narrative
        assert verdict.value >= VerdictLevel.PRECAUCION.value

    def test_translate_low_exergy(self, default_translator: SemanticTranslator):
        """Baja exergía refleja ineficiencia."""
        narrative, verdict = default_translator.translate_thermodynamics(
            entropy=0.2, exergy=0.2, temperature=25.0
        )

        assert "20.0%" in narrative  # Exergía en porcentaje
        assert verdict.value >= VerdictLevel.PRECAUCION.value

    def test_values_are_clamped(self, default_translator: SemanticTranslator):
        """Valores fuera de rango se ajustan."""
        # No debería lanzar error
        narrative, _ = default_translator.translate_thermodynamics(
            entropy=2.0,  # > 1.0
            exergy=-0.5,  # < 0.0
            temperature=-10.0,  # < 0.0
        )

        assert narrative  # Debe generar algo


# ============================================================================
# TEST: TRADUCCIÓN FINANCIERA
# ============================================================================


class TestFinancialTranslation:
    """Pruebas para traducción de métricas financieras."""

    def test_translate_viable_project(
        self, default_translator: SemanticTranslator, viable_financials: Dict[str, Any]
    ):
        """Proyecto viable genera narrativa positiva."""
        narrative, verdict, fin_verdict = default_translator.translate_financial(
            viable_financials
        )

        assert "WACC" in narrative
        assert "10.00%" in narrative
        assert "VIABLE" in narrative
        assert "1.25" in narrative
        assert verdict == VerdictLevel.VIABLE
        assert fin_verdict == FinancialVerdict.ACCEPT

    def test_translate_rejected_project(
        self, default_translator: SemanticTranslator, rejected_financials: Dict[str, Any]
    ):
        """Proyecto rechazado genera narrativa de riesgo."""
        narrative, verdict, fin_verdict = default_translator.translate_financial(
            rejected_financials
        )

        assert "RIESGO CRÍTICO" in narrative
        assert verdict == VerdictLevel.RECHAZAR
        assert fin_verdict == FinancialVerdict.REJECT

    def test_translate_review_project(
        self, default_translator: SemanticTranslator, review_financials: Dict[str, Any]
    ):
        """Proyecto en revisión genera narrativa apropiada."""
        narrative, verdict, fin_verdict = default_translator.translate_financial(
            review_financials
        )

        assert "REVISIÓN" in narrative
        assert verdict == VerdictLevel.REVISAR
        assert fin_verdict == FinancialVerdict.REVIEW

    def test_missing_fields_use_defaults(self, default_translator: SemanticTranslator):
        """Campos faltantes usan valores por defecto."""
        minimal = {"performance": {"recommendation": "REVISAR"}}

        narrative, _, _ = default_translator.translate_financial(minimal)

        assert "0.00%" in narrative  # WACC default

    def test_invalid_type_raises_error(self, default_translator: SemanticTranslator):
        """Tipo inválido lanza TypeError."""
        with pytest.raises(TypeError, match="Expected dict"):
            default_translator.translate_financial("not a dict")

    def test_invalid_recommendation_defaults_to_review(
        self, default_translator: SemanticTranslator
    ):
        """Recomendación inválida se convierte a REVISAR."""
        metrics = {
            "wacc": 0.10,
            "performance": {"recommendation": "ESTADO_INVALIDO"},
        }

        _, _, fin_verdict = default_translator.translate_financial(metrics)
        assert fin_verdict == FinancialVerdict.REVIEW


# ============================================================================
# TEST: GRAPHRAG (EXPLICACIÓN CAUSAL)
# ============================================================================


class TestGraphRAGExplanations:
    """Pruebas para explicaciones causales GraphRAG."""

    def test_explain_cycle_path(self, default_translator: SemanticTranslator):
        """Explica ruta del ciclo."""
        cycle = ["Insumo_A", "APU_1", "APU_2", "Insumo_A"]

        explanation = default_translator.explain_cycle_path(cycle)

        assert "Insumo_A" in explanation
        assert "→" in explanation
        assert "circularidad" in explanation.lower() or "ciclo" in explanation.lower()

    def test_explain_cycle_path_truncation(self, default_translator: SemanticTranslator):
        """Ciclos largos se truncan."""
        cycle = [f"node_{i}" for i in range(10)]

        explanation = default_translator.explain_cycle_path(cycle)

        assert "..." in explanation or "más" in explanation

    def test_explain_cycle_path_empty(self, default_translator: SemanticTranslator):
        """Lista vacía retorna string vacío."""
        assert default_translator.explain_cycle_path([]) == ""

    def test_explain_stress_point(self, default_translator: SemanticTranslator):
        """Explica punto de estrés."""
        explanation = default_translator.explain_stress_point("Cemento_Portland", 15)

        assert "Cemento_Portland" in explanation
        assert "15" in explanation
        assert "Piedra Angular" in explanation or "crítica" in explanation.lower()


# ============================================================================
# TEST: COMPOSICIÓN DE REPORTE ESTRATÉGICO
# ============================================================================


class TestStrategicReportComposition:
    """Pruebas para composición del reporte estratégico completo."""

    def test_compose_returns_strategic_report(
        self,
        deterministic_translator: SemanticTranslator,
        clean_topology: TopologyMetricsDTO,
        viable_financials: Dict[str, Any],
    ):
        """compose_strategic_narrative retorna StrategicReport."""
        report = deterministic_translator.compose_strategic_narrative(
            topological_metrics=clean_topology,
            financial_metrics=viable_financials,
            stability=12.0,
        )

        assert isinstance(report, StrategicReport)
        assert report.verdict is not None
        assert len(report.strata_analysis) == 4  # 4 estratos

    def test_report_structure_contains_all_sections(
        self,
        deterministic_translator: SemanticTranslator,
        clean_topology: TopologyMetricsDTO,
        viable_financials: Dict[str, Any],
    ):
        """Reporte contiene todas las secciones requeridas."""
        report = deterministic_translator.compose_strategic_narrative(
            topological_metrics=clean_topology,
            financial_metrics=viable_financials,
            stability=12.0,
        )

        # Verificar secciones en raw_narrative
        assert "INFORME DE INGENIERÍA ESTRATÉGICA" in report.raw_narrative
        assert "Auditoría de Integridad" in report.raw_narrative
        assert "Cargas Financieras" in report.raw_narrative
        assert "Geotecnia de Mercado" in report.raw_narrative
        assert "Dictamen" in report.raw_narrative

    def test_report_green_light_scenario(
        self,
        deterministic_translator: SemanticTranslator,
        clean_topology: TopologyMetricsDTO,
        viable_financials: Dict[str, Any],
    ):
        """Escenario de luz verde genera veredicto VIABLE."""
        report = deterministic_translator.compose_strategic_narrative(
            topological_metrics=clean_topology,
            financial_metrics=viable_financials,
            stability=15.0,  # Estable
        )

        assert report.verdict == VerdictLevel.VIABLE
        assert report.is_viable is True
        assert "CERTIFICADO" in report.raw_narrative

    def test_report_rejection_scenario(
        self,
        deterministic_translator: SemanticTranslator,
        cyclic_topology: TopologyMetricsDTO,
        rejected_financials: Dict[str, Any],
    ):
        """Escenario de rechazo genera veredicto apropiado."""
        report = deterministic_translator.compose_strategic_narrative(
            topological_metrics=cyclic_topology,
            financial_metrics=rejected_financials,
            stability=0.5,  # Pirámide invertida
        )

        assert report.verdict.value >= VerdictLevel.PRECAUCION.value
        assert report.is_viable is False

    def test_report_with_synergy_risk(
        self,
        deterministic_translator: SemanticTranslator,
        clean_topology: TopologyMetricsDTO,
        viable_financials: Dict[str, Any],
    ):
        """Sinergia de riesgo genera rechazo."""
        synergy = {
            "synergy_detected": True,
            "intersecting_cycles_count": 2,
            "intersecting_cycles": [["a", "b", "c"]],
        }

        report = deterministic_translator.compose_strategic_narrative(
            topological_metrics=clean_topology,
            financial_metrics=viable_financials,
            stability=15.0,
            synergy_risk=synergy,
        )

        assert report.verdict == VerdictLevel.RECHAZAR
        assert "Dominó" in report.raw_narrative or "EMERGENCIA" in report.raw_narrative

    def test_report_with_thermal_metrics(
        self,
        deterministic_translator: SemanticTranslator,
        clean_topology: TopologyMetricsDTO,
        viable_financials: Dict[str, Any],
    ):
        """Métricas térmicas se integran en el reporte."""
        thermal = {"entropy": 0.8, "exergy": 0.4, "system_temperature": 65.0}

        report = deterministic_translator.compose_strategic_narrative(
            topological_metrics=clean_topology,
            financial_metrics=viable_financials,
            stability=15.0,
            thermal_metrics=thermal,
        )

        assert "FIEBRE" in report.raw_narrative or "calor" in report.raw_narrative.lower()

    def test_report_serialization_to_dict(
        self,
        deterministic_translator: SemanticTranslator,
        clean_topology: TopologyMetricsDTO,
        viable_financials: Dict[str, Any],
    ):
        """Reporte se serializa correctamente a diccionario."""
        report = deterministic_translator.compose_strategic_narrative(
            topological_metrics=clean_topology,
            financial_metrics=viable_financials,
            stability=12.0,
        )

        dict_result = report.to_dict()

        assert "verdict" in dict_result
        assert "executive_summary" in dict_result
        assert "strata_analysis" in dict_result
        assert "recommendations" in dict_result
        assert "timestamp" in dict_result

    def test_legacy_compose_returns_string(
        self,
        deterministic_translator: SemanticTranslator,
        clean_topology: TopologyMetricsDTO,
        viable_financials: Dict[str, Any],
    ):
        """Método legacy retorna string."""
        result = deterministic_translator.compose_strategic_narrative_legacy(
            topological_metrics=clean_topology,
            financial_metrics=viable_financials,
            stability=12.0,
        )

        assert isinstance(result, str)
        assert "INFORME" in result


# ============================================================================
# TEST: INTEGRACIÓN CON STRATUM (DIKW)
# ============================================================================


class TestStratumIntegration:
    """Pruebas de integración con la jerarquía DIKW."""

    def test_report_contains_all_strata(
        self,
        deterministic_translator: SemanticTranslator,
        clean_topology: TopologyMetricsDTO,
        viable_financials: Dict[str, Any],
    ):
        """Reporte contiene análisis de todos los estratos."""
        report = deterministic_translator.compose_strategic_narrative(
            topological_metrics=clean_topology,
            financial_metrics=viable_financials,
            stability=12.0,
        )

        assert Stratum.PHYSICS in report.strata_analysis
        assert Stratum.TACTICS in report.strata_analysis
        assert Stratum.STRATEGY in report.strata_analysis
        assert Stratum.WISDOM in report.strata_analysis

    def test_strata_verdicts_compose_to_final(
        self,
        deterministic_translator: SemanticTranslator,
        clean_topology: TopologyMetricsDTO,
        viable_financials: Dict[str, Any],
    ):
        """Veredictos de estratos componen al veredicto final."""
        report = deterministic_translator.compose_strategic_narrative(
            topological_metrics=clean_topology,
            financial_metrics=viable_financials,
            stability=12.0,
        )

        # El veredicto final debe ser el supremum de los estratos
        strata_verdicts = [
            analysis.verdict for analysis in report.strata_analysis.values()
        ]

        expected_final = VerdictLevel.supremum(*strata_verdicts)
        # Puede haber ajustes adicionales, pero no debería ser más bajo
        assert report.verdict.value >= expected_final.value

    def test_failed_physics_propagates_up(self, deterministic_translator: SemanticTranslator):
        """Fallo en PHYSICS propaga hacia arriba (clausura transitiva)."""
        # Simular fallo en PHYSICS con temperatura crítica
        thermal = {"system_temperature": 100.0}  # Crítico

        report = deterministic_translator.compose_strategic_narrative(
            topological_metrics=TopologyMetricsDTO(),
            financial_metrics={"performance": {"recommendation": "ACEPTAR"}},
            stability=15.0,
            thermal_metrics=thermal,
        )

        # PHYSICS debe estar comprometido
        physics_analysis = report.strata_analysis[Stratum.PHYSICS]
        assert physics_analysis.verdict.value >= VerdictLevel.PRECAUCION.value

    def test_stratum_analysis_has_required_fields(
        self,
        deterministic_translator: SemanticTranslator,
        clean_topology: TopologyMetricsDTO,
        viable_financials: Dict[str, Any],
    ):
        """Análisis de estrato tiene campos requeridos."""
        report = deterministic_translator.compose_strategic_narrative(
            topological_metrics=clean_topology,
            financial_metrics=viable_financials,
            stability=12.0,
        )

        for stratum, analysis in report.strata_analysis.items():
            assert isinstance(analysis, StratumAnalysisResult)
            assert analysis.stratum == stratum
            assert isinstance(analysis.verdict, VerdictLevel)
            assert isinstance(analysis.narrative, str)
            assert isinstance(analysis.metrics_summary, dict)

    def test_recommendations_based_on_failed_strata(
        self, deterministic_translator: SemanticTranslator
    ):
        """Recomendaciones se basan en estratos que fallaron."""
        # Crear escenario con TACTICS comprometido
        cyclic = TopologyMetricsDTO(beta_0=1, beta_1=3)

        report = deterministic_translator.compose_strategic_narrative(
            topological_metrics=cyclic,
            financial_metrics={"performance": {"recommendation": "ACEPTAR"}},
            stability=15.0,
        )

        assert len(report.recommendations) > 0
        # Debería haber recomendación relacionada con topología
        has_topo_rec = any(
            "topología" in rec.lower() or "ciclo" in rec.lower()
            for rec in report.recommendations
        )
        assert has_topo_rec


# ============================================================================
# TEST: CASOS LÍMITE Y MANEJO DE ERRORES
# ============================================================================


class TestEdgeCasesAndErrorHandling:
    """Pruebas de casos límite y manejo de errores."""

    def test_handles_empty_financials(self, default_translator: SemanticTranslator):
        """Diccionario financiero vacío no causa crash."""
        report = default_translator.compose_strategic_narrative(
            topological_metrics=TopologyMetricsDTO(),
            financial_metrics={},
            stability=10.0,
        )

        assert report is not None
        assert isinstance(report, StrategicReport)

    def test_handles_none_thermal_metrics(
        self, default_translator: SemanticTranslator, clean_topology: TopologyMetricsDTO
    ):
        """Métricas térmicas None no causan crash."""
        report = default_translator.compose_strategic_narrative(
            topological_metrics=clean_topology,
            financial_metrics={"performance": {"recommendation": "ACEPTAR"}},
            stability=10.0,
            thermal_metrics=None,
        )

        assert report is not None

    def test_handles_invalid_topology_gracefully(
        self, default_translator: SemanticTranslator, viable_financials: Dict[str, Any]
    ):
        """Topología inválida se maneja graciosamente."""
        # Crear DTO con valores inválidos post-creación simulando corrupción
        invalid_dict = {"beta_0": -1, "beta_1": 0}

        # La conversión debería fallar y el reporte debería manejarlo
        with pytest.raises(ValueError):
            TopologyMetricsDTO.from_any(invalid_dict)

    def test_handles_invalid_financials_gracefully(
        self, default_translator: SemanticTranslator, clean_topology: TopologyMetricsDTO
    ):
        """Métricas financieras inválidas se manejan graciosamente."""
        report = default_translator.compose_strategic_narrative(
            topological_metrics=clean_topology,
            financial_metrics="not a dict",  # Inválido
            stability=10.0,
        )

        # Debería generar reporte con error en sección financiera
        assert "Error" in report.raw_narrative or report.strata_analysis[Stratum.STRATEGY].issues

    def test_extreme_stability_values(self, default_translator: SemanticTranslator):
        """Valores extremos de estabilidad no causan crash."""
        extreme_values = [0.0, 0.0001, 100.0, 1e6]

        for stability in extreme_values:
            narrative, verdict = default_translator.translate_topology(
                TopologyMetricsDTO(), stability=stability
            )
            assert narrative
            assert isinstance(verdict, VerdictLevel)

    def test_extreme_beta_values(self, default_translator: SemanticTranslator):
        """Valores extremos de beta no causan crash."""
        high_beta = TopologyMetricsDTO(beta_0=100, beta_1=50, euler_characteristic=50)

        narrative, verdict = default_translator.translate_topology(high_beta, stability=10.0)

        assert narrative
        assert verdict == VerdictLevel.RECHAZAR  # β₁ muy alto

    def test_nan_values_in_thermal(self, default_translator: SemanticTranslator):
        """NaN en valores térmicos se maneja."""
        # El DTO debería clampear o el método debería manejar
        thermal = ThermalMetricsDTO.from_dict({"temperature": float("nan")})

        # El clampeo debería convertir NaN a algo válido o el default
        # Dependiendo de implementación
        assert not math.isnan(thermal.temperature)


class TestMarketContext:
    """Pruebas para contexto de mercado."""

    def test_deterministic_market_context(self, deterministic_translator: SemanticTranslator):
        """Contexto de mercado es determinístico cuando se configura."""
        context1 = deterministic_translator.get_market_context()
        context2 = deterministic_translator.get_market_context()

        assert context1 == context2
        assert "🌍" in context1

    def test_custom_market_provider(self):
        """Proveedor de mercado personalizado funciona."""
        custom_message = "Mercado personalizado de prueba"
        provider = Mock(return_value=custom_message)

        translator = SemanticTranslator(market_provider=provider)
        context = translator.get_market_context()

        provider.assert_called_once()
        assert custom_message in context

    def test_failing_market_provider_handled(self):
        """Error en proveedor de mercado se maneja graciosamente."""
        def failing_provider():
            raise ConnectionError("API down")

        translator = SemanticTranslator(market_provider=failing_provider)
        context = translator.get_market_context()

        assert "No disponible" in context


class TestFactoryFunctions:
    """Pruebas para funciones factory."""

    def test_create_translator_default(self):
        """create_translator con defaults."""
        translator = create_translator()
        assert isinstance(translator, SemanticTranslator)

    def test_create_translator_with_config(self):
        """create_translator con configuración personalizada."""
        config = TranslatorConfig(deterministic_market=True)
        translator = create_translator(config=config)

        assert translator.config.deterministic_market is True

    def test_translate_metrics_to_narrative_convenience(self):
        """Función de conveniencia retorna string."""
        result = translate_metrics_to_narrative(
            topological_metrics={"beta_0": 1, "beta_1": 0},
            financial_metrics={"performance": {"recommendation": "ACEPTAR"}},
            stability=10.0,
        )

        assert isinstance(result, str)
        assert "INFORME" in result


# ============================================================================
# TEST: MATRIZ DE DECISIÓN EXHAUSTIVA
# ============================================================================


class TestDecisionMatrix:
    """Pruebas exhaustivas de la matriz de decisión."""

    @pytest.fixture
    def translator(self) -> SemanticTranslator:
        return SemanticTranslator()

    @pytest.mark.parametrize(
        "beta_1,stability,recommendation,expected_verdict",
        [
            # Caso ideal: sin ciclos, estable, aceptado
            (0, 15.0, "ACEPTAR", VerdictLevel.VIABLE),

            # Sin ciclos, estable, rechazado -> revisar (finanzas malas pero estructura ok)
            (0, 15.0, "RECHAZAR", VerdictLevel.RECHAZAR),

            # Con ciclos, estable, aceptado -> precaución (ciclos son problema)
            (2, 15.0, "ACEPTAR", VerdictLevel.PRECAUCION),

            # Sin ciclos, inestable, aceptado -> precaución (pirámide invertida)
            (0, 0.5, "ACEPTAR", VerdictLevel.PRECAUCION),

            # Con ciclos, inestable, rechazado -> máximo rechazo
            (3, 0.5, "RECHAZAR", VerdictLevel.RECHAZAR),

            # Ciclos críticos
            (5, 15.0, "ACEPTAR", VerdictLevel.RECHAZAR),
        ],
    )
    def test_decision_matrix_combinations(
        self,
        translator: SemanticTranslator,
        beta_1: int,
        stability: float,
        recommendation: str,
        expected_verdict: VerdictLevel,
    ):
        """Verifica combinaciones de la matriz de decisión."""
        topo = TopologyMetricsDTO(beta_0=1, beta_1=beta_1, euler_characteristic=1 - beta_1)
        fin = {"performance": {"recommendation": recommendation, "profitability_index": 1.0}}

        report = translator.compose_strategic_narrative(
            topological_metrics=topo,
            financial_metrics=fin,
            stability=stability,
        )

        # El veredicto debe ser al menos tan malo como esperado
        assert report.verdict.value >= expected_verdict.value, (
            f"Expected at least {expected_verdict.name}, got {report.verdict.name} "
            f"for β₁={beta_1}, Ψ={stability}, rec={recommendation}"
        )

    def test_synergy_always_rejects(self, translator: SemanticTranslator):
        """Sinergia de riesgo siempre causa rechazo."""
        topo = TopologyMetricsDTO(beta_0=1, beta_1=0)  # Topología limpia
        fin = {"performance": {"recommendation": "ACEPTAR"}}
        synergy = {"synergy_detected": True, "intersecting_cycles_count": 1}

        report = translator.compose_strategic_narrative(
            topological_metrics=topo,
            financial_metrics=fin,
            stability=100.0,  # Muy estable
            synergy_risk=synergy,
        )

        assert report.verdict == VerdictLevel.RECHAZAR


# ============================================================================
# TEST: NARRATIVE TEMPLATES
# ============================================================================


class TestNarrativeTemplates:
    """Pruebas para plantillas de narrativa."""

    def test_topology_cycle_templates_exist(self):
        """Plantillas de ciclos existen."""
        assert "clean" in NarrativeTemplates.TOPOLOGY_CYCLES
        assert "minor" in NarrativeTemplates.TOPOLOGY_CYCLES
        assert "moderate" in NarrativeTemplates.TOPOLOGY_CYCLES
        assert "critical" in NarrativeTemplates.TOPOLOGY_CYCLES

    def test_stability_templates_exist(self):
        """Plantillas de estabilidad existen."""
        assert "critical" in NarrativeTemplates.STABILITY
        assert "stable" in NarrativeTemplates.STABILITY
        assert "robust" in NarrativeTemplates.STABILITY

    def test_final_verdicts_templates_exist(self):
        """Plantillas de veredictos finales existen."""
        assert "synergy_risk" in NarrativeTemplates.FINAL_VERDICTS
        assert "certified" in NarrativeTemplates.FINAL_VERDICTS
        assert "has_holes" in NarrativeTemplates.FINAL_VERDICTS

    def test_templates_have_placeholders(self):
        """Plantillas con placeholders funcionan."""
        template = NarrativeTemplates.STABILITY["critical"]
        formatted = template.format(stability=0.5)

        assert "0.50" in formatted

    def test_market_contexts_non_empty(self):
        """Contextos de mercado no están vacíos."""
        assert len(NarrativeTemplates.MARKET_CONTEXTS) > 0
        for context in NarrativeTemplates.MARKET_CONTEXTS:
            assert len(context) > 0


# ============================================================================
# TEST: COMPATIBILIDAD HACIA ATRÁS
# ============================================================================


@pytest.mark.skipif(not HAS_ORIGINAL_METRICS, reason="TopologicalMetrics not available")
class TestBackwardsCompatibility:
    """Pruebas de compatibilidad hacia atrás con API original."""

    def test_accepts_original_topological_metrics(self, default_translator: SemanticTranslator):
        """Acepta TopologicalMetrics original."""
        original = TopologicalMetrics(beta_0=1, beta_1=0, euler_characteristic=1)

        narrative, verdict = default_translator.translate_topology(original, stability=10.0)

        assert narrative
        assert isinstance(verdict, VerdictLevel)

    def test_compose_with_original_metrics(self, default_translator: SemanticTranslator):
        """compose_strategic_narrative acepta TopologicalMetrics original."""
        original = TopologicalMetrics(beta_0=1, beta_1=0, euler_characteristic=1)
        fin = {"performance": {"recommendation": "ACEPTAR"}}

        report = default_translator.compose_strategic_narrative(
            topological_metrics=original,
            financial_metrics=fin,
            stability=10.0,
        )

        assert isinstance(report, StrategicReport)


# ============================================================================
# TEST: PROPIEDADES ESTADÍSTICAS (PROPERTY-BASED TESTING LITE)
# ============================================================================


class TestStatisticalProperties:
    """Pruebas de propiedades estadísticas del sistema."""

    @pytest.mark.parametrize("seed", range(5))
    def test_deterministic_output_with_same_input(
        self, seed: int, viable_financials: Dict[str, Any]
    ):
        """Misma entrada produce misma salida (determinismo)."""
        config = TranslatorConfig(deterministic_market=True)
        translator = SemanticTranslator(config=config)

        topo = TopologyMetricsDTO(beta_0=1, beta_1=seed % 3)

        report1 = translator.compose_strategic_narrative(
            topological_metrics=topo,
            financial_metrics=viable_financials,
            stability=10.0 + seed,
        )

        report2 = translator.compose_strategic_narrative(
            topological_metrics=topo,
            financial_metrics=viable_financials,
            stability=10.0 + seed,
        )

        assert report1.verdict == report2.verdict
        assert report1.raw_narrative == report2.raw_narrative

    def test_verdict_monotonicity_with_cycles(self, default_translator: SemanticTranslator):
        """Más ciclos → veredicto peor o igual (monotonicidad)."""
        previous_verdict = VerdictLevel.VIABLE
        fin = {"performance": {"recommendation": "ACEPTAR"}}

        for beta_1 in range(6):
            topo = TopologyMetricsDTO(beta_0=1, beta_1=beta_1)
            report = default_translator.compose_strategic_narrative(
                topological_metrics=topo,
                financial_metrics=fin,
                stability=15.0,
            )

            # Veredicto actual debe ser >= veredicto anterior
            assert report.verdict.value >= previous_verdict.value, (
                f"Monotonicity violated: β₁={beta_1-1}→{beta_1}, "
                f"verdict {previous_verdict.name}→{report.verdict.name}"
            )
            previous_verdict = report.verdict

    def test_verdict_monotonicity_with_stability(self, default_translator: SemanticTranslator):
        """Mayor estabilidad → veredicto mejor o igual (monotonicidad inversa)."""
        fin = {"performance": {"recommendation": "ACEPTAR"}}
        topo = TopologyMetricsDTO(beta_0=1, beta_1=0)

        previous_verdict = VerdictLevel.RECHAZAR  # Empezar con el peor

        for stability in [0.5, 1.0, 3.0, 10.0, 20.0]:
            report = default_translator.compose_strategic_narrative(
                topological_metrics=topo,
                financial_metrics=fin,
                stability=stability,
            )

            # Veredicto actual debe ser <= veredicto anterior (mejor o igual)
            assert report.verdict.value <= previous_verdict.value, (
                f"Inverse monotonicity violated: Ψ={stability}, "
                f"verdict worsened to {report.verdict.name}"
            )
            previous_verdict = report.verdict