# -*- coding: utf-8 -*-
"""
Suite de pruebas para el módulo de Traducción Semántica.

Valida la correcta traducción de métricas topológicas y financieras
a narrativas estratégicas de negocio.
"""

import pytest
from typing import Any, Dict
from unittest.mock import Mock

from agent.business_topology import TopologicalMetrics
from app.semantic_translator import (
    SemanticTranslator,
    StabilityThresholds,
    TopologicalThresholds,
    FinancialVerdict,
)


class TestSemanticTranslatorSetup:
    """Pruebas de inicialización y configuración del traductor."""

    def test_default_initialization(self):
        """Verifica inicialización con valores por defecto."""
        translator = SemanticTranslator()

        assert translator.stability_thresholds.critical == 1.0
        assert translator.stability_thresholds.solid == 10.0
        assert translator.topo_thresholds.connected_components_optimal == 1
        assert translator.topo_thresholds.cycles_optimal == 0

    def test_custom_thresholds_initialization(self):
        """Verifica inicialización con umbrales personalizados."""
        custom_stability = StabilityThresholds(critical=2.0, solid=15.0)
        custom_topo = TopologicalThresholds(
            connected_components_optimal=2,
            cycles_optimal=1
        )

        translator = SemanticTranslator(
            stability_thresholds=custom_stability,
            topo_thresholds=custom_topo
        )

        assert translator.stability_thresholds.critical == 2.0
        assert translator.stability_thresholds.solid == 15.0
        assert translator.topo_thresholds.connected_components_optimal == 2

    def test_market_provider_injection(self):
        """Verifica inyección de proveedor de mercado personalizado."""
        custom_context = "Tendencia personalizada de prueba"
        mock_provider = Mock(return_value=custom_context)

        translator = SemanticTranslator(market_provider=mock_provider)
        context = translator._get_market_context()

        mock_provider.assert_called_once()
        assert custom_context in context

    def test_random_seed_reproducibility(self):
        """Verifica que la semilla produce resultados reproducibles."""
        translator1 = SemanticTranslator(random_seed=42)
        translator2 = SemanticTranslator(random_seed=42)

        context1 = translator1._get_market_context()
        context2 = translator2._get_market_context()

        assert context1 == context2


class TestTopologyTranslation:
    """Pruebas para traducción de métricas topológicas."""

    @pytest.fixture
    def translator(self) -> SemanticTranslator:
        """Fixture del traductor con configuración estándar."""
        return SemanticTranslator(random_seed=42)

    @pytest.fixture
    def clean_metrics(self) -> TopologicalMetrics:
        """Métricas de un proyecto estructuralmente sano."""
        return TopologicalMetrics(beta_0=1, beta_1=0, euler_characteristic=1)

    @pytest.fixture
    def fragmented_metrics(self) -> TopologicalMetrics:
        """Métricas de un proyecto fragmentado sin ciclos."""
        return TopologicalMetrics(beta_0=3, beta_1=0, euler_characteristic=3)

    @pytest.fixture
    def cyclic_metrics(self) -> TopologicalMetrics:
        """Métricas de un proyecto con dependencias circulares."""
        return TopologicalMetrics(beta_0=1, beta_1=2, euler_characteristic=-1)

    def test_translate_topology_with_single_cycle(self, translator):
        """Verifica traducción con un único ciclo (singular)."""
        metrics = TopologicalMetrics(beta_0=1, beta_1=1, euler_characteristic=0)
        narrative = translator.translate_topology(metrics, stability=5.0)

        assert "Bloqueos Logísticos Detectados" in narrative
        assert "1 dependencia circular" in narrative
        assert "dependencias circulares" not in narrative  # Debe ser singular

    def test_translate_topology_with_multiple_cycles(self, translator, cyclic_metrics):
        """Verifica traducción con múltiples ciclos (plural)."""
        narrative = translator.translate_topology(cyclic_metrics, stability=5.0)

        assert "Bloqueos Logísticos Detectados" in narrative
        assert "2 dependencias circulares" in narrative

    def test_translate_topology_critical_cycles(self, translator):
        """Verifica severidad crítica cuando β₁ > 2."""
        metrics = TopologicalMetrics(beta_0=1, beta_1=5, euler_characteristic=-4)
        narrative = translator.translate_topology(metrics, stability=5.0)

        assert "críticos" in narrative
        assert "5 dependencias circulares" in narrative

    def test_translate_topology_clean_structure(self, translator, clean_metrics):
        """Verifica traducción de estructura limpia sin ciclos."""
        narrative = translator.translate_topology(clean_metrics, stability=25.0)

        assert "Flujo Logístico Optimizado" in narrative
        assert "Cohesión del Proyecto" in narrative
        assert "β₀ = 1" in narrative

    def test_translate_topology_fragmented(self, translator, fragmented_metrics):
        """Verifica detección de fragmentación (β₀ > 1)."""
        narrative = translator.translate_topology(fragmented_metrics, stability=5.0)

        assert "Fragmentación de Recursos" in narrative
        assert "3 islas de información" in narrative

    def test_translate_topology_severe_fragmentation(self, translator):
        """Verifica detección de fragmentación severa (β₀ > 3)."""
        metrics = TopologicalMetrics(beta_0=5, beta_1=0, euler_characteristic=5)
        narrative = translator.translate_topology(metrics, stability=5.0)

        assert "Severa" in narrative
        assert "5 islas" in narrative

    def test_translate_topology_empty_space(self, translator):
        """Verifica manejo de espacio vacío (β₀ = 0)."""
        metrics = TopologicalMetrics(beta_0=0, beta_1=0, euler_characteristic=0)
        narrative = translator.translate_topology(metrics, stability=5.0)

        assert "Estructura Vacía" in narrative
        assert "ausencia de componentes" in narrative

    @pytest.mark.parametrize("stability,expected_keyword", [
        (0.5, "Crítica"),
        (0.99, "Crítica"),
        (1.0, "Equilibrada"),
        (5.0, "Equilibrada"),
        (9.99, "Equilibrada"),
        (10.0, "Sólida"),
        (25.0, "Sólida"),
        (100.0, "Sólida"),
    ])
    def test_translate_stability_thresholds(
        self,
        translator,
        clean_metrics,
        stability: float,
        expected_keyword: str
    ):
        """Verifica clasificación correcta según umbrales de estabilidad."""
        narrative = translator.translate_topology(clean_metrics, stability=stability)

        assert expected_keyword in narrative
        assert f"Ψ = {stability:.2f}" in narrative


class TestTopologyValidation:
    """Pruebas de validación de métricas topológicas."""

    @pytest.fixture
    def translator(self) -> SemanticTranslator:
        return SemanticTranslator()

    def test_validate_negative_beta_0_raises_error(self, translator):
        """β₀ negativo debe lanzar ValueError."""
        invalid_metrics = TopologicalMetrics(beta_0=-1, beta_1=0, euler_characteristic=-1)

        with pytest.raises(ValueError, match="β₀ debe ser no-negativo"):
            translator.translate_topology(invalid_metrics, stability=5.0)

    def test_validate_negative_beta_1_raises_error(self, translator):
        """β₁ negativo debe lanzar ValueError."""
        invalid_metrics = TopologicalMetrics(beta_0=1, beta_1=-2, euler_characteristic=3)

        with pytest.raises(ValueError, match="β₁ debe ser no-negativo"):
            translator.translate_topology(invalid_metrics, stability=5.0)

    def test_validate_negative_stability_raises_error(self, translator):
        """Estabilidad negativa debe lanzar ValueError."""
        valid_metrics = TopologicalMetrics(beta_0=1, beta_1=0, euler_characteristic=1)

        with pytest.raises(ValueError, match="Estabilidad Ψ debe ser no-negativa"):
            translator.translate_topology(valid_metrics, stability=-0.5)

    def test_validate_invalid_metrics_type_raises_error(self, translator):
        """Tipo incorrecto de métricas debe lanzar ValueError."""
        with pytest.raises(ValueError, match="Se esperaba TopologicalMetrics"):
            translator.translate_topology({"beta_0": 1, "beta_1": 0}, stability=5.0)


class TestFinancialTranslation:
    """Pruebas para traducción de métricas financieras."""

    @pytest.fixture
    def translator(self) -> SemanticTranslator:
        return SemanticTranslator()

    @pytest.fixture
    def viable_metrics(self) -> Dict[str, Any]:
        """Métricas de un proyecto financieramente viable."""
        return {
            "wacc": 0.12,
            "var": 50000.0,
            "contingency": {"recommended": 60000.0},
            "performance": {
                "recommendation": "ACEPTAR",
                "profitability_index": 1.25
            }
        }

    @pytest.fixture
    def rejected_metrics(self) -> Dict[str, Any]:
        """Métricas de un proyecto rechazado."""
        return {
            "wacc": 0.18,
            "var": 100000.0,
            "contingency": {"recommended": 150000.0},
            "performance": {
                "recommendation": "RECHAZAR",
                "profitability_index": 0.75
            }
        }

    @pytest.fixture
    def review_metrics(self) -> Dict[str, Any]:
        """Métricas que requieren revisión."""
        return {
            "wacc": 0.10,
            "var": 1000.0,
            "contingency": {"recommended": 1500.0},
            "performance": {"recommendation": "REVISAR"}
        }

    def test_translate_financial_success(self, translator, viable_metrics):
        """Verifica traducción de proyecto viable."""
        narrative = translator.translate_financial(viable_metrics)

        assert "Costo de Oportunidad del Capital (WACC)" in narrative
        assert "12.00%" in narrative
        assert "Exposición al Riesgo Financiero" in narrative
        assert "$60,000.00" in narrative
        assert "FINANCIERAMENTE VIABLE" in narrative
        assert "1.25" in narrative

    def test_translate_financial_rejection(self, translator, rejected_metrics):
        """Verifica traducción de proyecto rechazado."""
        narrative = translator.translate_financial(rejected_metrics)

        assert "RIESGOS CRÍTICOS" in narrative
        assert "0.75" in narrative
        assert "reestructurar los costos" in narrative

    def test_translate_financial_review(self, translator, review_metrics):
        """Verifica traducción de proyecto en revisión."""
        narrative = translator.translate_financial(review_metrics)

        assert "revisión manual profunda" in narrative
        assert "inconsistencias" in narrative

    def test_translate_wacc_elevated(self, translator):
        """Verifica clasificación de WACC elevado (> 15%)."""
        metrics = {
            "wacc": 0.20,
            "contingency": {"recommended": 1000.0},
            "performance": {"recommendation": "REVISAR"}
        }
        narrative = translator.translate_financial(metrics)

        assert "elevado para el sector" in narrative

    def test_translate_wacc_competitive(self, translator):
        """Verifica clasificación de WACC competitivo (< 5%)."""
        metrics = {
            "wacc": 0.03,
            "contingency": {"recommended": 1000.0},
            "performance": {"recommendation": "ACEPTAR", "profitability_index": 1.5}
        }
        narrative = translator.translate_financial(metrics)

        assert "competitivo" in narrative

    def test_translate_zero_contingency(self, translator):
        """Verifica mensaje cuando contingencia es cero."""
        metrics = {
            "wacc": 0.10,
            "contingency": {"recommended": 0.0},
            "performance": {"recommendation": "REVISAR"}
        }
        narrative = translator.translate_financial(metrics)

        assert "No se ha calculado contingencia" in narrative


class TestFinancialValidation:
    """Pruebas de validación de métricas financieras."""

    @pytest.fixture
    def translator(self) -> SemanticTranslator:
        return SemanticTranslator()

    def test_validate_invalid_metrics_type_raises_error(self, translator):
        """Tipo incorrecto de métricas debe lanzar ValueError."""
        with pytest.raises(ValueError, match="Se esperaba dict"):
            translator.translate_financial("not a dict")

    def test_validate_invalid_wacc_type_raises_error(self, translator):
        """WACC no numérico debe lanzar ValueError."""
        metrics = {
            "wacc": "invalid",
            "performance": {"recommendation": "REVISAR"}
        }

        with pytest.raises(ValueError, match="WACC debe ser numérico"):
            translator.translate_financial(metrics)

    def test_missing_wacc_uses_default(self, translator, caplog):
        """WACC ausente usa valor por defecto con warning."""
        metrics = {
            "contingency": {"recommended": 1000.0},
            "performance": {"recommendation": "REVISAR"}
        }

        narrative = translator.translate_financial(metrics)

        assert "0.00%" in narrative
        assert "WACC no especificado" in caplog.text

    def test_unknown_recommendation_defaults_to_review(self, translator, caplog):
        """Recomendación desconocida se convierte a REVISAR."""
        metrics = {
            "wacc": 0.10,
            "contingency": {"recommended": 1000.0},
            "performance": {"recommendation": "ESTADO_INVALIDO"}
        }

        narrative = translator.translate_financial(metrics)

        assert "revisión manual" in narrative
        assert "no reconocida" in caplog.text

    def test_malformed_performance_handled_gracefully(self, translator):
        """Performance malformado no causa crash."""
        metrics = {
            "wacc": 0.10,
            "contingency": {"recommended": 1000.0},
            "performance": "not a dict"
        }

        # No debe lanzar excepción
        narrative = translator.translate_financial(metrics)
        assert "revisión manual" in narrative

    def test_malformed_contingency_handled_gracefully(self, translator):
        """Contingency malformado no causa crash."""
        metrics = {
            "wacc": 0.10,
            "contingency": "not a dict",
            "performance": {"recommendation": "REVISAR"}
        }

        narrative = translator.translate_financial(metrics)
        assert "No se ha calculado contingencia" in narrative


class TestStrategicNarrative:
    """Pruebas para composición del reporte estratégico completo."""

    @pytest.fixture
    def translator(self) -> SemanticTranslator:
        return SemanticTranslator(random_seed=42)

    @pytest.fixture
    def clean_topo_metrics(self) -> TopologicalMetrics:
        return TopologicalMetrics(beta_0=1, beta_1=0, euler_characteristic=1)

    @pytest.fixture
    def cyclic_topo_metrics(self) -> TopologicalMetrics:
        return TopologicalMetrics(beta_0=1, beta_1=2, euler_characteristic=-1)

    @pytest.fixture
    def accept_fin_metrics(self) -> Dict[str, Any]:
        return {
            "wacc": 0.10,
            "var": 1000.0,
            "contingency": {"recommended": 1500.0},
            "performance": {
                "recommendation": "ACEPTAR",
                "profitability_index": 1.2
            }
        }

    @pytest.fixture
    def reject_fin_metrics(self) -> Dict[str, Any]:
        return {
            "wacc": 0.15,
            "var": 50000.0,
            "contingency": {"recommended": 75000.0},
            "performance": {
                "recommendation": "RECHAZAR",
                "profitability_index": 0.6
            }
        }

    @pytest.fixture
    def review_fin_metrics(self) -> Dict[str, Any]:
        return {
            "wacc": 0.10,
            "var": 1000.0,
            "contingency": {"recommended": 1500.0},
            "performance": {"recommendation": "REVISAR"}
        }

    def test_compose_strategic_narrative_structure(
        self,
        translator,
        clean_topo_metrics,
        accept_fin_metrics
    ):
        """Verifica estructura completa del reporte."""
        report = translator.compose_strategic_narrative(
            clean_topo_metrics,
            accept_fin_metrics,
            stability=12.0
        )

        # Secciones obligatorias
        assert "## 🏗️ INFORME DE INTELIGENCIA ESTRATÉGICA" in report
        assert "### 1. Salud Estructural y Operativa" in report
        assert "### 2. Análisis de Viabilidad Económica" in report
        assert "### 3. Inteligencia de Mercado" in report
        assert "### 💡 Recomendación Estratégica" in report

    def test_compose_strategic_narrative_green_light(
        self,
        translator,
        clean_topo_metrics,
        accept_fin_metrics
    ):
        """Verifica luz verde: sin ciclos + aceptado."""
        report = translator.compose_strategic_narrative(
            clean_topo_metrics,
            accept_fin_metrics,
            stability=12.0
        )

        assert "LUZ VERDE" in report
        assert "β₁ = 0" in report
        assert "coherencia técnica" in report

    def test_compose_strategic_narrative_review_status(
        self,
        translator,
        clean_topo_metrics,
        review_fin_metrics
    ):
        """Verifica estado REVISAR: sin ciclos + revisar."""
        report = translator.compose_strategic_narrative(
            clean_topo_metrics,
            review_fin_metrics,
            stability=12.0
        )

        assert "EVALUACIÓN INCOMPLETA" in report
        assert "LUZ VERDE" not in report
        # assert "certeza financiera" in narrative or "certeza financiera" in report # narrative var not defined here
        assert "certeza financiera" in report

    def test_compose_strategic_narrative_financial_rejection(
        self,
        translator,
        clean_topo_metrics,
        reject_fin_metrics
    ):
        """Verifica rechazo financiero: sin ciclos + rechazado."""
        report = translator.compose_strategic_narrative(
            clean_topo_metrics,
            reject_fin_metrics,
            stability=12.0
        )

        assert "REVISIÓN FINANCIERA" in report
        assert "estructura técnica es sólida" in report
        assert "números no cierran" in report

    def test_compose_strategic_narrative_technical_issues(
        self,
        translator,
        cyclic_topo_metrics,
        accept_fin_metrics
    ):
        """Verifica cautela: ciclos + aceptado financieramente."""
        report = translator.compose_strategic_narrative(
            cyclic_topo_metrics,
            accept_fin_metrics,
            stability=12.0
        )

        assert "PROCEDER CON CAUTELA" in report
        assert "2 ciclo(s)" in report
        assert "litigios" in report

    def test_compose_strategic_narrative_total_failure(
        self,
        translator,
        cyclic_topo_metrics,
        reject_fin_metrics
    ):
        """Verifica fallo total: ciclos + rechazado."""
        report = translator.compose_strategic_narrative(
            cyclic_topo_metrics,
            reject_fin_metrics,
            stability=12.0
        )

        assert "ACCIÓN INMEDIATA REQUERIDA" in report
        assert "inviable técnica y financieramente" in report
        assert "Detener procesos" in report

    def test_compose_strategic_narrative_cycles_with_review(
        self,
        translator,
        cyclic_topo_metrics,
        review_fin_metrics
    ):
        """Verifica auditoría: ciclos + revisar."""
        report = translator.compose_strategic_narrative(
            cyclic_topo_metrics,
            review_fin_metrics,
            stability=12.0
        )

        assert "AUDITORÍA REQUERIDA" in report
        assert "corrección topológica" in report


class TestStrategicNarrativeErrorHandling:
    """Pruebas de manejo de errores en composición estratégica."""

    @pytest.fixture
    def translator(self) -> SemanticTranslator:
        return SemanticTranslator(random_seed=42)

    def test_handles_invalid_topology_gracefully(self, translator):
        """Errores topológicos se capturan sin crash."""
        invalid_topo = TopologicalMetrics(beta_0=-1, beta_1=0, euler_characteristic=-1)
        valid_fin = {
            "wacc": 0.10,
            "contingency": {"recommended": 1000.0},
            "performance": {"recommendation": "ACEPTAR", "profitability_index": 1.2}
        }

        report = translator.compose_strategic_narrative(
            invalid_topo,
            valid_fin,
            stability=5.0
        )

        assert "ANÁLISIS INCOMPLETO" in report
        assert "Error en análisis estructural" in report

    def test_handles_invalid_financials_gracefully(self, translator):
        """Errores financieros se capturan sin crash."""
        valid_topo = TopologicalMetrics(beta_0=1, beta_1=0, euler_characteristic=1)
        invalid_fin = "not a dict"

        report = translator.compose_strategic_narrative(
            valid_topo,
            invalid_fin,
            stability=5.0
        )

        assert "ANÁLISIS INCOMPLETO" in report
        assert "Error en análisis financiero" in report


class TestMarketContext:
    """Pruebas para obtención de contexto de mercado."""

    def test_market_context_contains_emoji(self):
        """Contexto de mercado incluye emoji de mundo."""
        translator = SemanticTranslator(random_seed=42)
        context = translator._get_market_context()

        assert "🌍" in context
        assert "Contexto de Mercado" in context

    def test_market_provider_error_handled(self, caplog):
        """Error en proveedor de mercado se maneja graciosamente."""
        def failing_provider():
            raise ConnectionError("API no disponible")

        translator = SemanticTranslator(market_provider=failing_provider)
        context = translator._get_market_context()

        assert "No disponible temporalmente" in context
        assert "Error obteniendo contexto" in caplog.text

    def test_market_context_variety(self):
        """Diferentes semillas producen diferentes contextos."""
        contexts = set()
        for seed in range(10):
            translator = SemanticTranslator(random_seed=seed)
            contexts.add(translator._get_market_context())

        # Debería haber variedad (al menos 2 diferentes)
        assert len(contexts) >= 2


class TestFinalAdviceDecisionMatrix:
    """Pruebas exhaustivas de la matriz de decisión del consejo final."""

    @pytest.fixture
    def translator(self) -> SemanticTranslator:
        return SemanticTranslator()

    @pytest.mark.parametrize("beta_1,recommendation,expected_keywords", [
        # Sin ciclos
        (0, "ACEPTAR", ["LUZ VERDE", "coherencia técnica"]),
        (0, "RECHAZAR", ["REVISIÓN FINANCIERA", "números no cierran"]),
        (0, "REVISAR", ["EVALUACIÓN INCOMPLETA", "certeza financiera"]),
        # Con ciclos
        (1, "ACEPTAR", ["PROCEDER CON CAUTELA", "litigios"]),
        (1, "RECHAZAR", ["ACCIÓN INMEDIATA REQUERIDA", "inviable"]),
        (1, "REVISAR", ["AUDITORÍA REQUERIDA", "corrección topológica"]),
        # Múltiples ciclos
        (3, "RECHAZAR", ["ACCIÓN INMEDIATA", "3 dependencia"]),
    ])
    def test_decision_matrix_coverage(
        self,
        translator,
        beta_1: int,
        recommendation: str,
        expected_keywords: list
    ):
        """Verifica todas las combinaciones de la matriz de decisión."""
        topo = TopologicalMetrics(
            beta_0=1,
            beta_1=beta_1,
            euler_characteristic=1 - beta_1
        )
        fin = {"performance": {"recommendation": recommendation}}

        advice = translator._generate_final_advice(topo, fin)

        for keyword in expected_keywords:
            assert keyword in advice, f"'{keyword}' no encontrado en: {advice}"
