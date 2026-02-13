"""
Suite de pruebas para el módulo de Traducción Semántica.

Valida:
1. Propiedades algebraicas del Lattice de Veredictos
2. Correcta traducción de métricas topológicas y financieras
3. Normalización de datos vía DTOs
4. Integración con jerarquía DIKW/Stratum
5. Composición de reportes estratégicos
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

from app.telemetry_schemas import (
    PhysicsMetrics,
    TopologicalMetrics,
    ControlMetrics,
    ThermodynamicMetrics,
)

from app.schemas import Stratum


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
def clean_topology() -> TopologicalMetrics:
    """Topología limpia: conexa, sin ciclos."""
    return TopologicalMetrics(beta_0=1, beta_1=0, euler_characteristic=1)


@pytest.fixture
def cyclic_topology() -> TopologicalMetrics:
    """Topología con ciclos."""
    return TopologicalMetrics(beta_0=1, beta_1=2, euler_characteristic=-1)


@pytest.fixture
def fragmented_topology() -> TopologicalMetrics:
    """Topología fragmentada."""
    return TopologicalMetrics(beta_0=3, beta_1=0, euler_characteristic=3)


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
# TEST: SCHEMAS
# ============================================================================


class TestTelemetrySchemas:
    """Pruebas para esquemas de telemetría."""

    def test_topological_metrics_defaults(self):
        """Valores por defecto de TopologicalMetrics."""
        metrics = TopologicalMetrics()
        assert metrics.beta_0 == 1
        assert metrics.beta_1 == 0
        assert metrics.fiedler_value == 1.0

    def test_thermodynamic_metrics_defaults(self):
        """Valores por defecto de ThermodynamicMetrics."""
        metrics = ThermodynamicMetrics()
        assert metrics.entropy == 0.0
        assert metrics.exergy == 1.0
        assert metrics.system_temperature == 25.0

    def test_physics_metrics_defaults(self):
        """Valores por defecto de PhysicsMetrics."""
        metrics = PhysicsMetrics()
        assert metrics.gyroscopic_stability == 1.0
        assert metrics.saturation == 0.0

    def test_control_metrics_defaults(self):
        """Valores por defecto de ControlMetrics."""
        metrics = ControlMetrics()
        assert metrics.is_stable is True
        assert metrics.phase_margin_deg == 45.0


# ============================================================================
# TEST: PROPIEDADES ALGEBRAICAS DEL LATTICE
# ============================================================================


class TestLatticeAlgebraicProperties:
    """Verifica las propiedades algebraicas del lattice (VerdictLevel, ≤, ⊔, ⊓)."""

    def test_lattice_has_bottom_and_top(self):
        assert VerdictLevel.bottom() == VerdictLevel.VIABLE
        assert VerdictLevel.top() == VerdictLevel.RECHAZAR

    def test_supremum_takes_worst_case(self):
        result = VerdictLevel.supremum(
            VerdictLevel.VIABLE,
            VerdictLevel.CONDICIONAL,
            VerdictLevel.RECHAZAR,
        )
        assert result == VerdictLevel.RECHAZAR


# ============================================================================
# TEST: TRADUCCIÓN DE TOPOLOGÍA
# ============================================================================


class TestTopologyTranslation:
    """Pruebas para traducción de métricas topológicas."""

    def test_translate_clean_topology(
        self, default_translator: SemanticTranslator, clean_topology: TopologicalMetrics
    ):
        """Topología limpia genera narrativa positiva."""
        narrative, verdict = default_translator.translate_topology(
            clean_topology, stability=15.0
        )

        assert "Integridad Estructural" in narrative or "Genus 0" in narrative
        assert verdict == VerdictLevel.VIABLE

    def test_translate_cyclic_topology(
        self, default_translator: SemanticTranslator, cyclic_topology: TopologicalMetrics
    ):
        """Topología con ciclos genera advertencias."""
        narrative, verdict = default_translator.translate_topology(
            cyclic_topology, stability=15.0
        )

        assert "socavones" in narrative.lower() or "ciclos" in narrative.lower() or "bucles" in narrative.lower()
        assert verdict.value >= VerdictLevel.CONDICIONAL.value

    def test_translate_accepts_dict_input(self, default_translator: SemanticTranslator):
        """Acepta diccionario como input."""
        dict_input = {"beta_0": 1, "beta_1": 0, "euler_characteristic": 1}

        narrative, verdict = default_translator.translate_topology(
            dict_input, stability=10.0
        )

        assert "Integridad" in narrative
        assert verdict == VerdictLevel.VIABLE

    def test_translate_negative_stability_raises_error(
        self, default_translator: SemanticTranslator, clean_topology: TopologicalMetrics
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
        narrative, verdict = default_translator.translate_thermodynamics(
            entropy=0.2, exergy=0.9, temperature=15.0
        )
        assert "Estable" in narrative or "❄️" in narrative
        assert verdict == VerdictLevel.VIABLE

    def test_translate_critical_temperature(self, default_translator: SemanticTranslator):
        narrative, verdict = default_translator.translate_thermodynamics(
            entropy=0.5, exergy=0.5, temperature=80.0
        )
        assert "FUSIÓN" in narrative or "crítica" in narrative.lower()
        assert verdict == VerdictLevel.RECHAZAR


# ============================================================================
# TEST: TRADUCCIÓN DE FÍSICA Y CONTROL
# ============================================================================


class TestPhysicsTranslation:
    """Pruebas para traducción de métricas físicas."""

    def test_translate_unstable_gyroscope(self, default_translator: SemanticTranslator):
        """Nutación crítica genera rechazo."""
        physics = PhysicsMetrics(gyroscopic_stability=0.2)  # < 0.3
        result = default_translator._analyze_physics_stratum(
            ThermodynamicMetrics(), stability=10.0, physics=physics
        )
        assert "NUTACIÓN" in result.narrative
        assert result.verdict == VerdictLevel.RECHAZAR

    def test_translate_unstable_laplace(self, default_translator: SemanticTranslator):
        """Inestabilidad de Laplace (RHP) genera rechazo."""
        control = ControlMetrics(is_stable=False)
        result = default_translator._analyze_physics_stratum(
            ThermodynamicMetrics(), stability=10.0, control=control
        )
        assert "DIVERGENCIA" in result.narrative or "Divergencia" in result.narrative
        assert result.verdict == VerdictLevel.RECHAZAR


# ============================================================================
# TEST: COMPOSICIÓN DE REPORTE ESTRATÉGICO
# ============================================================================


class TestStrategicReportComposition:
    """Pruebas para composición del reporte estratégico completo."""

    def test_compose_returns_strategic_report(
        self,
        deterministic_translator: SemanticTranslator,
        clean_topology: TopologicalMetrics,
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

    def test_report_green_light_scenario(
        self,
        deterministic_translator: SemanticTranslator,
        clean_topology: TopologicalMetrics,
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

    def test_report_with_thermal_metrics(
        self,
        deterministic_translator: SemanticTranslator,
        clean_topology: TopologicalMetrics,
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


# ============================================================================
# TEST: INTEGRACIÓN CON STRATUM (DIKW)
# ============================================================================


class TestStratumIntegration:
    """Pruebas de integración con la jerarquía DIKW."""

    def test_failed_physics_propagates_up(self, deterministic_translator: SemanticTranslator):
        """Fallo en PHYSICS propaga hacia arriba (clausura transitiva)."""
        # Simular fallo en PHYSICS con temperatura crítica
        thermal = {"system_temperature": 100.0}  # Crítico

        report = deterministic_translator.compose_strategic_narrative(
            topological_metrics=TopologicalMetrics(),
            financial_metrics={"performance": {"recommendation": "ACEPTAR"}},
            stability=15.0,
            thermal_metrics=thermal,
        )

        # PHYSICS debe estar comprometido
        physics_analysis = report.strata_analysis[Stratum.PHYSICS]
        assert physics_analysis.verdict.value >= VerdictLevel.PRECAUCION.value


# ============================================================================
# TEST: PROPIEDADES ESTADÍSTICAS
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

        topo = TopologicalMetrics(beta_0=1, beta_1=seed % 3)

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
