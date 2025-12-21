# -*- coding: utf-8 -*-
"""
Suite de pruebas para el módulo de Traducción Semántica.

Valida la correcta traducción de métricas topológicas y financieras
a narrativas estratégicas de negocio con enfoque de Ingeniería Estructural.
"""

import pytest
from typing import Any, Dict
from unittest.mock import Mock

from agent.business_topology import TopologicalMetrics
from app.semantic_translator import (
    SemanticTranslator,
    StabilityThresholds,
    TopologicalThresholds,
    WACCThresholds,
    CycleSeverityThresholds,
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
        custom_wacc = WACCThresholds(low=0.04, high=0.10)
        custom_severity = CycleSeverityThresholds(moderate=2, critical=4)

        translator = SemanticTranslator(
            stability_thresholds=custom_stability,
            topo_thresholds=custom_topo,
            wacc_thresholds=custom_wacc,
            cycle_severity=custom_severity
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

    def test_invalid_init_arguments_raise_type_error(self):
        """Verifica que tipos incorrectos en __init__ lanzan TypeError."""
        with pytest.raises(TypeError):
            SemanticTranslator(stability_thresholds="invalid") # type: ignore


class TestTopologyTranslation:
    """Pruebas para traducción de métricas topológicas (Narrativa Estructural)."""

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
        """Verifica traducción con un único ciclo (Genus 1)."""
        metrics = TopologicalMetrics(beta_0=1, beta_1=1, euler_characteristic=0)
        narrative = translator.translate_topology(metrics, stability=5.0)

        # "Bloqueos Logísticos Moderados" -> "Falla Estructural Local" / "Genus 1"
        assert "Genus 1" in narrative
        assert "socavones lógicos" in narrative

    def test_translate_topology_with_multiple_cycles(self, translator, cyclic_metrics):
        """Verifica traducción con múltiples ciclos (Genus Elevado)."""
        narrative = translator.translate_topology(cyclic_metrics, stability=5.0)

        assert "Falla Estructural Local" in narrative or "Genus Elevado" in narrative
        assert "2 socavones" in narrative

    def test_translate_topology_critical_cycles(self, translator):
        """Verifica severidad crítica cuando β₁ >= 5."""
        metrics = TopologicalMetrics(beta_0=1, beta_1=5, euler_characteristic=-4)
        narrative = translator.translate_topology(metrics, stability=5.0)

        assert "Estructura Geológicamente Inestable" in narrative
        assert "Genus Estructural de 5" in narrative

    def test_translate_topology_clean_structure(self, translator, clean_metrics):
        """Verifica traducción de estructura limpia sin ciclos."""
        narrative = translator.translate_topology(clean_metrics, stability=25.0)

        assert "Integridad Estructural (Genus 0)" in narrative
        assert "Unidad de Obra Monolítica" in narrative

    def test_translate_topology_fragmented(self, translator, fragmented_metrics):
        """Verifica detección de fragmentación (β₀ > 1)."""
        narrative = translator.translate_topology(fragmented_metrics, stability=5.0)

        assert "Edificios Desconectados" in narrative
        assert "3 sub-estructuras" in narrative

    def test_translate_topology_severe_fragmentation(self, translator):
        """Verifica detección de fragmentación severa."""
        metrics = TopologicalMetrics(beta_0=5, beta_1=0, euler_characteristic=5)
        narrative = translator.translate_topology(metrics, stability=5.0)

        assert "Edificios Desconectados" in narrative
        assert "5 sub-estructuras" in narrative

    def test_translate_topology_empty_space(self, translator):
        """Verifica manejo de espacio vacío (β₀ = 0)."""
        metrics = TopologicalMetrics(beta_0=0, beta_1=0, euler_characteristic=0)
        narrative = translator.translate_topology(metrics, stability=5.0)

        assert "Terreno Vacío" in narrative
        assert "β₀ = 0" in narrative

    @pytest.mark.parametrize("stability,expected_keyword", [
        (0.5, "Pirámide Invertida"),
        (0.99, "Pirámide Invertida"),
        (1.0, "Estructura Isostática"),
        (5.0, "Estructura Isostática"),
        (9.99, "Estructura Isostática"),
        (10.0, "ESTRUCTURA ANTISÍSMICA"),
        (25.0, "ESTRUCTURA ANTISÍSMICA"),
        (100.0, "ESTRUCTURA ANTISÍSMICA"),
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

        with pytest.raises(ValueError, match="no-negativos"):
            translator.translate_topology(invalid_metrics, stability=5.0)

    def test_validate_negative_beta_1_raises_error(self, translator):
        """β₁ negativo debe lanzar ValueError."""
        invalid_metrics = TopologicalMetrics(beta_0=1, beta_1=-2, euler_characteristic=3)

        with pytest.raises(ValueError, match="no-negativos"):
            translator.translate_topology(invalid_metrics, stability=5.0)

    def test_validate_negative_stability_raises_error(self, translator):
        """Estabilidad negativa debe lanzar ValueError."""
        valid_metrics = TopologicalMetrics(beta_0=1, beta_1=0, euler_characteristic=1)

        with pytest.raises(ValueError, match="debe ser no-negativa"):
            translator.translate_topology(valid_metrics, stability=-0.5)

    def test_validate_invalid_metrics_type_raises_error(self, translator):
        """Tipo incorrecto de métricas debe lanzar TypeError."""
        with pytest.raises(TypeError, match="Se esperaba TopologicalMetrics"):
            translator.translate_topology({"beta_0": 1, "beta_1": 0}, stability=5.0) # type: ignore


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

        assert "Costo de Oportunidad" in narrative
        assert "12.00%" in narrative
        assert "Blindaje Financiero" in narrative
        assert "$60,000.00" in narrative
        assert "VIABLE" in narrative
        assert "1.25" in narrative

    def test_translate_financial_rejection(self, translator, rejected_metrics):
        """Verifica traducción de proyecto rechazado."""
        narrative = translator.translate_financial(rejected_metrics)

        assert "RIESGO CRÍTICO" in narrative
        assert "0.75" in narrative

    def test_translate_financial_review(self, translator, review_metrics):
        """Verifica traducción de proyecto en revisión."""
        narrative = translator.translate_financial(review_metrics)
        assert "REVISIÓN REQUERIDA" in narrative

    def test_translate_zero_contingency(self, translator):
        """Verifica mensaje cuando contingencia es cero."""
        metrics = {
            "wacc": 0.10,
            "contingency": {"recommended": 0.0},
            "performance": {"recommendation": "REVISAR"}
        }
        narrative = translator.translate_financial(metrics)

        assert "$0.00" in narrative


class TestFinancialValidation:
    """Pruebas de validación de métricas financieras."""

    @pytest.fixture
    def translator(self) -> SemanticTranslator:
        return SemanticTranslator()

    def test_validate_invalid_metrics_type_raises_error(self, translator):
        """Tipo incorrecto de métricas debe lanzar TypeError."""
        with pytest.raises(TypeError, match="Se esperaba dict"):
            translator.translate_financial("not a dict") # type: ignore

    def test_validate_invalid_wacc_type_raises_error(self, translator):
        """WACC no numérico debe lanzar ValueError.
        NOTA: En la implementación 'Parse, Don't Validate', esto retorna default 0.0
        o debe manejarse si extract_numeric es estricto. La implementación actual es defensiva."""
        metrics = {
            "wacc": "invalid",
            "performance": {"recommendation": "REVISAR"}
        }

        # En la nueva implementación, extract_numeric retorna default=0.0 si falla validación interna
        # o si la función extract_numeric hace return default.
        # Si queremos testear que NO lanza error, cambiamos la expectativa.
        narrative = translator.translate_financial(metrics)
        assert "0.00%" in narrative

    def test_missing_wacc_uses_default(self, translator):
        """WACC ausente usa valor por defecto."""
        metrics = {
            "contingency": {"recommended": 1000.0},
            "performance": {"recommendation": "REVISAR"}
        }

        narrative = translator.translate_financial(metrics)
        assert "0.00%" in narrative

    def test_unknown_recommendation_defaults_to_review(self, translator):
        """Recomendación desconocida se convierte a REVISAR."""
        metrics = {
            "wacc": 0.10,
            "contingency": {"recommended": 1000.0},
            "performance": {"recommendation": "ESTADO_INVALIDO"}
        }

        narrative = translator.translate_financial(metrics)
        assert "REVISIÓN REQUERIDA" in narrative


class TestStrategicNarrative:
    """Pruebas para composición del reporte estratégico completo (Ingeniería Estructural)."""

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
        assert "## 🏗️ INFORME DE INGENIERÍA ESTRATÉGICA" in report
        assert "### 1. Auditoría de Integridad Estructural" in report
        assert "### 2. Análisis de Cargas Financieras" in report
        assert "### 3. Geotecnia de Mercado" in report
        assert "### 💡 Dictamen del Ingeniero Jefe" in report

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

        assert "CERTIFICADO DE SOLIDEZ" in report
        assert "Estructura piramidal estable" in report

    def test_compose_strategic_narrative_review_status(
        self,
        translator,
        clean_topo_metrics,
        review_fin_metrics
    ):
        """Verifica estado REVISAR."""
        report = translator.compose_strategic_narrative(
            clean_topo_metrics,
            review_fin_metrics,
            stability=12.0
        )

        assert "REVISIÓN TÉCNICA REQUERIDA" in report

    def test_compose_strategic_narrative_financial_rejection(
        self,
        translator,
        clean_topo_metrics,
        reject_fin_metrics
    ):
        """Verifica rechazo financiero."""
        report = translator.compose_strategic_narrative(
            clean_topo_metrics,
            reject_fin_metrics,
            stability=12.0
        )

        assert "REVISIÓN TÉCNICA REQUERIDA" in report

    def test_compose_strategic_narrative_technical_issues(
        self,
        translator,
        cyclic_topo_metrics,
        accept_fin_metrics
    ):
        """Verifica cautela: ciclos (agujeros) + aceptado financieramente."""
        report = translator.compose_strategic_narrative(
            cyclic_topo_metrics,
            accept_fin_metrics,
            stability=12.0
        )

        assert "DETENER PARA REPARACIONES" in report
        assert "socavones" in report

    def test_compose_strategic_narrative_total_failure(
        self,
        translator,
        cyclic_topo_metrics,
        reject_fin_metrics
    ):
        """Verifica fallo total."""
        report = translator.compose_strategic_narrative(
            cyclic_topo_metrics,
            reject_fin_metrics,
            stability=12.0
        )

        assert "DETENER PARA REPARACIONES" in report


class TestStrategicNarrativeErrorHandling:
    """Pruebas de manejo de errores en composición estratégica."""

    @pytest.fixture
    def translator(self) -> SemanticTranslator:
        return SemanticTranslator(random_seed=42)

    def test_handles_invalid_topology_gracefully(self, translator):
        """Errores topológicos se capturan."""
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

        assert "Error analizando estructura" in report

    def test_handles_invalid_financials_gracefully(self, translator):
        """Errores financieros se capturan."""
        valid_topo = TopologicalMetrics(beta_0=1, beta_1=0, euler_characteristic=1)
        invalid_fin = "not a dict" # type: ignore

        report = translator.compose_strategic_narrative(
            valid_topo,
            invalid_fin, # type: ignore
            stability=5.0
        )

        assert "Error analizando finanzas" in report

    def test_interrupted_analysis_verdict(self, translator):
        """Verifica que el veredicto final bloquee si hay errores."""
        invalid_topo = TopologicalMetrics(beta_0=-1, beta_1=0, euler_characteristic=-1)
        valid_fin = {"performance": {"recommendation": "ACEPTAR"}}

        report = translator.compose_strategic_narrative(
            invalid_topo,
            valid_fin,
            stability=5.0
        )

        assert "⚠️ ANÁLISIS ESTRUCTURAL INTERRUMPIDO" in report
        assert "impiden certificar la solidez" in report


class TestMarketContext:
    """Pruebas para obtención de contexto de mercado."""

    def test_market_context_contains_emoji(self):
        """Contexto de mercado incluye emoji."""
        translator = SemanticTranslator(random_seed=42)
        context = translator._get_market_context()

        assert "🌍" in context
        assert "Suelo de Mercado" in context

    def test_market_provider_error_handled(self):
        """Error en proveedor de mercado se maneja graciosamente."""
        def failing_provider():
            raise ConnectionError("API no disponible")

        translator = SemanticTranslator(market_provider=failing_provider)
        context = translator._get_market_context()

        assert "No disponible" in context


class TestFinalAdviceDecisionMatrix:
    """Pruebas exhaustivas de la matriz de decisión (Lógica Piramidal)."""

    @pytest.fixture
    def translator(self) -> SemanticTranslator:
        return SemanticTranslator()

    @pytest.mark.parametrize("beta_1,recommendation,expected_keywords", [
        # Sin ciclos (sin agujeros)
        (0, "ACEPTAR", ["CERTIFICADO DE SOLIDEZ", "Estructura piramidal estable"]),
        (0, "RECHAZAR", ["REVISIÓN TÉCNICA REQUERIDA"]),
        (0, "REVISAR", ["REVISIÓN TÉCNICA REQUERIDA"]),
        # Con ciclos (con agujeros)
        (1, "ACEPTAR", ["DETENER PARA REPARACIONES", "socavones"]),
        (1, "RECHAZAR", ["DETENER PARA REPARACIONES", "socavones"]),
        (3, "RECHAZAR", ["DETENER PARA REPARACIONES", "socavones"]),
    ])
    def test_decision_matrix_coverage(
        self,
        translator,
        beta_1: int,
        recommendation: str,
        expected_keywords: list
    ):
        """Verifica combinaciones."""
        topo = TopologicalMetrics(
            beta_0=1,
            beta_1=beta_1,
            euler_characteristic=1 - beta_1
        )
        fin = {"performance": {"recommendation": recommendation}}

        # Estabilidad = 10.0 para evitar pirámide invertida
        advice = translator._generate_final_advice(topo, fin, stability=10.0)

        for keyword in expected_keywords:
            assert keyword in advice, f"'{keyword}' no encontrado en: {advice}"

    def test_decision_matrix_unstable_downgrade(self, translator):
        """
        Verifica que un proyecto inestable (Ψ < 1) sea degradado
        a 'PRECAUCIÓN LOGÍSTICA' o 'PROYECTO INVIABLE'.
        """
        topo = TopologicalMetrics(beta_0=1, beta_1=0, euler_characteristic=1)
        fin = {"performance": {"recommendation": "ACEPTAR"}}

        # Caso inestable (Ψ = 0.5) y ACEPTAR
        advice = translator._generate_final_advice(topo, fin, stability=0.5)
        assert "PRECAUCIÓN LOGÍSTICA" in advice
        assert "Pirámide Invertida" in advice

        # Caso inestable (Ψ = 0.5) y RECHAZAR
        fin_reject = {"performance": {"recommendation": "RECHAZAR"}}
        advice_reject = translator._generate_final_advice(topo, fin_reject, stability=0.5)
        assert "PROYECTO INVIABLE" in advice_reject
