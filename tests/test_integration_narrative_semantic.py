"""
Suite de integración: TelemetryNarrator ↔ SemanticTranslator.

Fundamentación matemática:
──────────────────────────

1. Morfismo de retículos (lattice morphism):
   SeverityLevel (L_S) y VerdictLevel (L_V) son retículos finitos
   totalmente ordenados. Existe un morfismo monótono φ: L_S → L_V
   que preserva el orden:

       a ≤_S b  ⟹  φ(a) ≤_V φ(b)

   Dado que |L_S| = 3 y |L_V| = 5, NO existe isomorfismo
   (cardinalidades distintas), pero sí una inmersión de orden.

   Referencia: [1] Davey & Priestley, "Introduction to Lattices
   and Order", 2nd Ed., Cambridge, 2002.

2. Conexión de Galois:
   Las funciones α: L_S → L_V (abstracción) y γ: L_V → L_S
   (concretización) forman una conexión de Galois si:

       α(s) ≤_V v  ⟺  s ≤_S γ(v)

   Esto formaliza la relación entre severidad técnica y
   veredicto estratégico.

3. Consistencia jerárquica DIKW:
   Los estratos forman una cadena:
       PHYSICS ≤ TACTICS ≤ STRATEGY ≤ WISDOM

   La propiedad de monotonicidad negativa exige:
       Fail(estrato_k) ⟹ ∀j > k: severity(j) ≥ severity(k)

   Es decir, los problemas no se "curan" al subir en la jerarquía.

4. Composición de reportes:
   El flujo completo es un funtor compuesto:

       F = Translator ∘ Extract ∘ Narrator: TelemetryContext → StrategicReport

   La coherencia exige que F preserve el orden de severidad.

Referencias:
    [1] Davey & Priestley, "Introduction to Lattices and Order", 2002.
    [2] Cousot & Cousot, "Abstract Interpretation", ACM POPL, 1977.
    [3] Ackoff, R. "From Data to Wisdom", J. Applied Systems Analysis, 1989.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pytest

from app.core.telemetry_narrative import (
    SeverityLevel,
    TelemetryNarrator,
    StratumTopology,
    NarratorConfig,
    create_narrator,
)
from app.wisdom.semantic_translator import (
    VerdictLevel,
    StrategicReport,
    SemanticTranslator,
    TranslatorConfig,
    create_translator,
)
from app.core.telemetry_schemas import TopologicalMetrics
from app.core.telemetry import TelemetryContext, StepStatus
from app.core.schemas import Stratum


# =============================================================================
# CONSTANTES
# =============================================================================

# Mapeo canónico de inmersión φ: SeverityLevel → VerdictLevel.
# Preserva el orden: OPTIMO ≤ ADVERTENCIA ≤ CRITICO
# se mapea a: VIABLE ≤ PRECAUCION ≤ RECHAZAR.
_SEVERITY_TO_VERDICT: Dict[SeverityLevel, VerdictLevel] = {
    SeverityLevel.OPTIMO: VerdictLevel.VIABLE,
    SeverityLevel.ADVERTENCIA: VerdictLevel.REVISAR,
    SeverityLevel.CRITICO: VerdictLevel.RECHAZAR,
}

# Mapeo adjunto derecho γ: VerdictLevel → SeverityLevel (concretización).
# Para la conexión de Galois: α(s) ≤ v ⟺ s ≤ γ(v).
_VERDICT_TO_SEVERITY: Dict[VerdictLevel, SeverityLevel] = {
    VerdictLevel.VIABLE: SeverityLevel.OPTIMO,
    VerdictLevel.CONDICIONAL: SeverityLevel.OPTIMO,
    VerdictLevel.REVISAR: SeverityLevel.ADVERTENCIA,
    VerdictLevel.PRECAUCION: SeverityLevel.ADVERTENCIA,
    VerdictLevel.RECHAZAR: SeverityLevel.CRITICO,
}

# Orden numérico de severity para comparaciones.
_SEVERITY_ORDER: Dict[str, int] = {
    "OPTIMO": 0,
    "ADVERTENCIA": 1,
    "CRITICO": 2,
}

# Pasos de pipeline y sus estratos para tests de consistencia.
_PHYSICS_STEPS: List[str] = ["load_data", "merge_data", "flux_condenser"]
_TACTICS_STEPS: List[str] = ["calculate_costs", "materialization"]
_STRATEGY_STEPS: List[str] = ["financial_analysis", "business_topology"]
_WISDOM_STEPS: List[str] = ["build_output", "response_preparation"]


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================


def _get_severity_rank(severity_name: str) -> int:
    """
    Retorna el rango numérico de un nivel de severidad.

    OPTIMO = 0, ADVERTENCIA = 1, CRITICO = 2.

    Parameters
    ----------
    severity_name : str
        Nombre del nivel de severidad.

    Returns
    -------
    int
        Rango numérico. Default 2 (CRITICO) para desconocidos.
    """
    return _SEVERITY_ORDER.get(severity_name, 2)


def _create_telemetry_context_with_steps(
    step_configs: List[Dict[str, Any]],
) -> TelemetryContext:
    """
    Crea un TelemetryContext con pasos configurados.

    Parameters
    ----------
    step_configs : List[Dict]
        Lista de configuraciones de pasos:
        - "name": str — nombre del paso
        - "status": StepStatus — estado (default SUCCESS)
        - "errors": List[Dict] — errores opcionales
        - "metrics": Dict — métricas opcionales {(group, key): value}

    Returns
    -------
    TelemetryContext
        Contexto configurado.
    """
    context = TelemetryContext()

    for config in step_configs:
        with context.span(config["name"]) as span:
            if config.get("status") == StepStatus.FAILURE:
                span.status = StepStatus.FAILURE
            elif config.get("status") == StepStatus.WARNING:
                span.status = StepStatus.WARNING

            for error in config.get("errors", []):
                span.errors.append(error)

            for (group, key), value in config.get("metrics", {}).items():
                context.record_metric(group, key, value)

    return context


def _create_successful_context() -> TelemetryContext:
    """Crea un contexto de telemetría exitoso con pasos básicos."""
    return _create_telemetry_context_with_steps([
        {"name": "load_data"},
        {"name": "calculate_costs"},
        {"name": "financial_analysis"},
        {"name": "build_output"},
    ])


def _create_physics_failure_context() -> TelemetryContext:
    """Crea un contexto con fallo en PHYSICS."""
    return _create_telemetry_context_with_steps([
        {
            "name": "load_data",
            "status": StepStatus.FAILURE,
            "errors": [{"message": "Data corruption", "type": "DataError"}],
        },
    ])


def _create_tactics_failure_context() -> TelemetryContext:
    """Crea un contexto con fallo en TACTICS."""
    return _create_telemetry_context_with_steps([
        {"name": "load_data"},
        {
            "name": "calculate_costs",
            "status": StepStatus.FAILURE,
            "errors": [{"message": "Ciclo detectado", "type": "CycleError"}],
        },
    ])


def _assert_narrative_contains_any(
    narrative: str,
    terms: List[str],
    context: str = "",
) -> None:
    """
    Verifica que la narrativa contiene al menos uno de los términos.

    Parameters
    ----------
    narrative : str
        Texto a buscar.
    terms : List[str]
        Términos candidatos (case-insensitive, búsqueda parcial).
    context : str
        Descripción para mensajes de error.
    """
    narrative_lower = narrative.lower()
    found = [t for t in terms if t.lower() in narrative_lower]

    assert len(found) > 0, (
        f"[{context}] Narrativa no contiene ninguno de: {terms}.\n"
        f"Narrativa (500 chars): '{narrative_lower[:500]}...'"
    )


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def narrator() -> TelemetryNarrator:
    """Narrador de telemetría con configuración por defecto."""
    return TelemetryNarrator()


@pytest.fixture
def translator() -> SemanticTranslator:
    """Traductor semántico en modo determinista."""
    return SemanticTranslator(
        config=TranslatorConfig(deterministic_market=True)
    )


@pytest.fixture
def clean_topology() -> TopologicalMetrics:
    """
    Topología limpia: grafo conexo sin ciclos.

    β₀ = 1 (una componente conexa)
    β₁ = 0 (sin ciclos)
    χ = β₀ - β₁ = 1
    """
    return TopologicalMetrics(
        beta_0=1, beta_1=0, euler_characteristic=1
    )


@pytest.fixture
def cyclic_topology() -> TopologicalMetrics:
    """
    Topología con ciclos: grafo conexo con 3 ciclos independientes.

    β₀ = 1, β₁ = 3
    χ = 1 - 3 = -2
    """
    return TopologicalMetrics(
        beta_0=1, beta_1=3, euler_characteristic=-2
    )


@pytest.fixture
def viable_financials() -> Dict[str, Any]:
    """Métricas financieras que indican proyecto viable."""
    return {
        "wacc": 0.10,
        "contingency": {"recommended": 5000.0},
        "performance": {
            "recommendation": "ACEPTAR",
            "profitability_index": 1.25,
        },
    }


@pytest.fixture
def rejected_financials() -> Dict[str, Any]:
    """Métricas financieras que indican rechazo."""
    return {
        "wacc": 0.18,
        "contingency": {"recommended": 50000.0},
        "performance": {
            "recommendation": "RECHAZAR",
            "profitability_index": 0.7,
        },
    }


# =============================================================================
# TEST SUITE 1: MORFISMO DE RETÍCULOS
# =============================================================================


class TestLatticeEmbedding:
    """
    Verifica la inmersión monótona φ: SeverityLevel → VerdictLevel.

    Como |L_S| = 3 y |L_V| = 5, no existe isomorfismo.
    Se verifica en cambio que φ es un morfismo de orden:
        a ≤_S b ⟹ φ(a) ≤_V φ(b)

    Y que los elementos extremales se preservan:
        φ(⊥_S) = ⊥_V  (mejor caso → mejor caso)
        φ(⊤_S) = ⊤_V  (peor caso → peor caso)
    """

    def test_cardinalities_differ(self) -> None:
        """
        |L_S| ≠ |L_V|, por lo tanto no existe isomorfismo.

        SeverityLevel: {OPTIMO, ADVERTENCIA, CRITICO} → |L_S| = 3
        VerdictLevel: {VIABLE, CONDICIONAL, REVISAR, PRECAUCION, RECHAZAR} → |L_V| = 5
        """
        assert len(SeverityLevel) == 3
        assert len(VerdictLevel) == 5
        assert len(SeverityLevel) != len(VerdictLevel), (
            "Si las cardinalidades fueran iguales, "
            "el isomorfismo sería posible."
        )

    def test_embedding_preserves_order(self) -> None:
        """
        φ preserva el orden: a ≤_S b ⟹ φ(a) ≤_V φ(b).

        Se verifica exhaustivamente para todos los pares en L_S.
        """
        severity_list = list(SeverityLevel)

        for a in severity_list:
            for b in severity_list:
                if a.value <= b.value:
                    phi_a = _SEVERITY_TO_VERDICT[a]
                    phi_b = _SEVERITY_TO_VERDICT[b]
                    assert phi_a.value <= phi_b.value, (
                        f"Ruptura de monotonicidad: "
                        f"φ({a.name}) = {phi_a.name} (val={phi_a.value}), "
                        f"φ({b.name}) = {phi_b.name} (val={phi_b.value}). "
                        f"Pero {a.name} ≤ {b.name} exige "
                        f"φ({a.name}) ≤ φ({b.name})."
                    )

    def test_bottom_maps_to_bottom(self) -> None:
        """
        φ(⊥_S) = ⊥_V: el mejor caso de severidad se mapea al
        mejor caso de veredicto.

        ⊥_S = OPTIMO, ⊥_V = VIABLE.
        """
        bottom_severity = min(SeverityLevel, key=lambda s: s.value)
        bottom_verdict = min(VerdictLevel, key=lambda v: v.value)

        assert bottom_severity == SeverityLevel.OPTIMO
        assert bottom_verdict == VerdictLevel.VIABLE
        assert _SEVERITY_TO_VERDICT[bottom_severity] == bottom_verdict

    def test_top_maps_to_top(self) -> None:
        """
        φ(⊤_S) = ⊤_V: el peor caso de severidad se mapea al
        peor caso de veredicto.

        ⊤_S = CRITICO, ⊤_V = RECHAZAR.
        """
        top_severity = max(SeverityLevel, key=lambda s: s.value)
        top_verdict = max(VerdictLevel, key=lambda v: v.value)

        assert top_severity == SeverityLevel.CRITICO
        assert top_verdict == VerdictLevel.RECHAZAR
        assert _SEVERITY_TO_VERDICT[top_severity] == top_verdict

    def test_galois_connection_property(self) -> None:
        """
        Verifica la propiedad de conexión de Galois:

            α(s) ≤_V v  ⟺  s ≤_S γ(v)

        donde α = _SEVERITY_TO_VERDICT, γ = _VERDICT_TO_SEVERITY.

        Esta propiedad formaliza que α y γ son adjuntos y que la
        relación entre severidad y veredicto es coherente en ambas
        direcciones.
        """
        for s in SeverityLevel:
            for v in VerdictLevel:
                alpha_s = _SEVERITY_TO_VERDICT[s]
                gamma_v = _VERDICT_TO_SEVERITY[v]

                lhs = alpha_s.value <= v.value
                rhs = s.value <= gamma_v.value

                assert lhs == rhs, (
                    f"Conexión de Galois violada: "
                    f"α({s.name})={alpha_s.name} ≤ {v.name} es {lhs}, "
                    f"pero {s.name} ≤ γ({v.name})={gamma_v.name} es {rhs}."
                )


# =============================================================================
# TEST SUITE 2: CONSISTENCIA DE STRATUM
# =============================================================================


class TestStratumConsistency:
    """Verifica que ambos módulos usan Stratum consistentemente."""

    def test_narrator_covers_all_strata(self) -> None:
        """El narrador tiene mapeo para todos los estratos."""
        narrator_strata = set(StratumTopology.HIERARCHY.keys())

        for stratum in Stratum:
            assert stratum in narrator_strata, (
                f"Stratum.{stratum.name} no está en StratumTopology.HIERARCHY."
            )

    def test_evaluation_order_base_to_top(self) -> None:
        """
        El orden de evaluación va de base (PHYSICS) a cima (WISDOM).

        Esto es necesario para la clausura transitiva:
        si PHYSICS falla, los estratos superiores heredan el fallo.
        """
        order = StratumTopology.EVALUATION_ORDER

        assert order[0] == Stratum.PHYSICS, (
            f"Primer estrato debe ser PHYSICS, es {order[0]}."
        )
        assert order[-1] == Stratum.WISDOM, (
            f"Último estrato debe ser WISDOM, es {order[-1]}."
        )

    def test_step_mapping_covers_critical_steps(self) -> None:
        """
        El mapeo de pasos cubre todas las etapas del pipeline.
        """
        mapping = StratumTopology.DEFAULT_STEP_MAPPING

        for step in _PHYSICS_STEPS:
            assert mapping.get(step) == Stratum.PHYSICS, (
                f"Paso '{step}' debería mapear a PHYSICS."
            )

        for step in _TACTICS_STEPS:
            assert mapping.get(step) == Stratum.TACTICS, (
                f"Paso '{step}' debería mapear a TACTICS."
            )

        for step in _STRATEGY_STEPS:
            assert mapping.get(step) == Stratum.STRATEGY, (
                f"Paso '{step}' debería mapear a STRATEGY."
            )

        for step in _WISDOM_STEPS:
            assert mapping.get(step) == Stratum.WISDOM, (
                f"Paso '{step}' debería mapear a WISDOM."
            )

    def test_translator_produces_strata_analysis(
        self,
        translator: SemanticTranslator,
        clean_topology: TopologicalMetrics,
        viable_financials: Dict,
    ) -> None:
        """
        El translator produce análisis para cada estrato.
        """
        report = translator.compose_strategic_narrative(
            topological_metrics=clean_topology,
            financial_metrics=viable_financials,
            stability=10.0,
        )

        for stratum in Stratum:
            assert stratum in report.strata_analysis, (
                f"Stratum.{stratum.name} ausente en strata_analysis."
            )


# =============================================================================
# TEST SUITE 3: FLUJO END-TO-END
# =============================================================================


@pytest.mark.integration
class TestEndToEndFlow:
    """
    Tests del flujo completo:
        TelemetryContext → Narrator → (extracción) → Translator → StrategicReport

    Verifica que el funtor compuesto F = Translator ∘ Extract ∘ Narrator
    preserva la coherencia de severidad/veredicto.
    """

    def test_successful_pipeline(
        self,
        narrator: TelemetryNarrator,
        translator: SemanticTranslator,
    ) -> None:
        """
        Pipeline exitoso: todos los pasos pasan → veredicto positivo.
        """
        context = _create_successful_context()

        telemetry_report = narrator.summarize_execution(context)
        assert telemetry_report["verdict_code"] == "APPROVED"

        strategic_report = translator.compose_strategic_narrative(
            topological_metrics=TopologicalMetrics(beta_0=1, beta_1=0),
            financial_metrics={
                "performance": {
                    "recommendation": "ACEPTAR",
                    "profitability_index": 1.2,
                },
            },
            stability=15.0,
        )

        assert strategic_report.verdict == VerdictLevel.VIABLE, (
            f"Pipeline exitoso debería dar VIABLE, "
            f"obtenido {strategic_report.verdict.name}."
        )

    def test_physics_failure_propagates(
        self,
        narrator: TelemetryNarrator,
        translator: SemanticTranslator,
    ) -> None:
        """
        Fallo en PHYSICS → ambos módulos rechazan.
        """
        context = _create_physics_failure_context()

        telemetry_report = narrator.summarize_execution(context)
        assert "PHYSICS" in telemetry_report["verdict_code"], (
            f"Narrator debería reportar fallo en PHYSICS, "
            f"verdict_code = {telemetry_report['verdict_code']}."
        )

        strategic_report = translator.compose_strategic_narrative(
            topological_metrics=TopologicalMetrics(beta_0=0, beta_1=0),
            financial_metrics={
                "performance": {"recommendation": "RECHAZAR"},
            },
            stability=0.5,
        )

        assert strategic_report.verdict == VerdictLevel.RECHAZAR, (
            f"Fallo en PHYSICS debería dar RECHAZAR, "
            f"obtenido {strategic_report.verdict.name}."
        )

    def test_tactics_failure_detects_cycles(
        self,
        narrator: TelemetryNarrator,
        translator: SemanticTranslator,
    ) -> None:
        """
        Fallo en TACTICS (ciclos) → narrativa menciona ciclos.
        """
        context = _create_tactics_failure_context()

        telemetry_report = narrator.summarize_execution(context)
        assert "TACTICS" in telemetry_report["verdict_code"]

        strategic_report = translator.compose_strategic_narrative(
            topological_metrics=TopologicalMetrics(beta_0=1, beta_1=3),
            financial_metrics={
                "performance": {"recommendation": "REVISAR"},
            },
            stability=10.0,
        )

        _assert_narrative_contains_any(
            strategic_report.raw_narrative,
            ["socavón", "ciclo", "agujero", "β₁", "homolog"],
            "tactics-cycles",
        )

    def test_thermal_metrics_flow(
        self,
        narrator: TelemetryNarrator,
        translator: SemanticTranslator,
    ) -> None:
        """
        Métricas térmicas fluyen correctamente entre módulos.
        """
        context = _create_telemetry_context_with_steps([
            {
                "name": "flux_condenser",
                "metrics": {
                    ("flux_condenser", "system_temperature"): 65.0,
                    ("flux_condenser", "entropy"): 0.8,
                },
            },
        ])

        narrator.summarize_execution(context)

        strategic_report = translator.compose_strategic_narrative(
            topological_metrics=TopologicalMetrics(),
            financial_metrics={
                "performance": {"recommendation": "ACEPTAR"},
            },
            stability=10.0,
            thermal_metrics={
                "system_temperature": 65.0,
                "entropy": 0.8,
            },
        )

        _assert_narrative_contains_any(
            strategic_report.raw_narrative,
            ["fiebre", "calor", "térmic", "temperatura"],
            "thermal-flow",
        )


# =============================================================================
# TEST SUITE 4: COHERENCIA DE NARRATIVAS
# =============================================================================


@pytest.mark.integration
class TestNarrativeCoherence:
    """
    Verifica coherencia semántica entre narrativas de ambos módulos.

    Invariantes:
        (C1) Pipeline exitoso → ambas narrativas positivas
        (C2) Pipeline fallido → ambas narrativas negativas
        (C3) Monotonicidad: más problemas → peor veredicto
    """

    def test_success_narratives_are_positive(
        self,
        narrator: TelemetryNarrator,
        translator: SemanticTranslator,
    ) -> None:
        """
        Invariante (C1): pipeline exitoso → narrativas positivas.
        """
        context = _create_successful_context()

        telemetry_report = narrator.summarize_execution(context)
        strategic_report = translator.compose_strategic_narrative(
            topological_metrics=TopologicalMetrics(),
            financial_metrics={
                "performance": {
                    "recommendation": "ACEPTAR",
                    "profitability_index": 1.5,
                },
            },
            stability=20.0,
        )

        _assert_narrative_contains_any(
            telemetry_report.get("executive_summary", ""),
            ["✅", "CERTIFICADO", "aprobad", "exitoso"],
            "narrator-positive",
        )
        assert strategic_report.verdict.is_positive, (
            f"Translator debería dar veredicto positivo, "
            f"obtenido {strategic_report.verdict.name}."
        )

    def test_failure_narratives_are_negative(
        self,
        narrator: TelemetryNarrator,
        translator: SemanticTranslator,
    ) -> None:
        """
        Invariante (C2): pipeline fallido → narrativas negativas.
        """
        context = _create_physics_failure_context()

        telemetry_report = narrator.summarize_execution(context)
        strategic_report = translator.compose_strategic_narrative(
            topological_metrics=TopologicalMetrics(beta_0=0, beta_1=5),
            financial_metrics={
                "performance": {"recommendation": "RECHAZAR"},
            },
            stability=0.5,
        )

        _assert_narrative_contains_any(
            telemetry_report.get("verdict_code", ""),
            ["REJECTED", "RECHAZADO", "PHYSICS"],
            "narrator-negative",
        )
        assert strategic_report.verdict.is_negative, (
            f"Translator debería dar veredicto negativo, "
            f"obtenido {strategic_report.verdict.name}."
        )

    def test_monotonicity_more_cycles_worse_verdict(
        self,
        translator: SemanticTranslator,
    ) -> None:
        """
        Invariante (C3): más ciclos (β₁ creciente) → veredicto
        no mejora (monótonamente peor o igual).

        Para β₁ = 0, 1, 2, 3:
            verdict(β₁=0) ≤ verdict(β₁=1) ≤ verdict(β₁=2) ≤ verdict(β₁=3)
        """
        previous_verdict_value: int = 0

        for beta_1 in range(4):
            report = translator.compose_strategic_narrative(
                topological_metrics=TopologicalMetrics(
                    beta_0=1, beta_1=beta_1
                ),
                financial_metrics={
                    "performance": {"recommendation": "REVISAR"},
                },
                stability=10.0,
            )

            assert report.verdict.value >= previous_verdict_value, (
                f"Monotonicidad violada: "
                f"verdict(β₁={beta_1}) = {report.verdict.name} "
                f"(val={report.verdict.value}) < "
                f"previous (val={previous_verdict_value})."
            )
            previous_verdict_value = report.verdict.value


# =============================================================================
# TEST SUITE 5: PROPAGACIÓN DE ERRORES ENTRE MÓDULOS
# =============================================================================


@pytest.mark.integration
class TestErrorPropagation:
    """
    Verifica que errores en un módulo no causan crasheo en el otro.

    Cada módulo debe ser resiliente a:
    - Datos incompletos del otro módulo
    - Valores None
    - Formatos inesperados
    """

    def test_translator_works_without_narrator_data(
        self,
        translator: SemanticTranslator,
    ) -> None:
        """
        El translator funciona con datos mínimos, incluso si
        el narrator no pudo ejecutarse.
        """
        report = translator.compose_strategic_narrative(
            topological_metrics=TopologicalMetrics(),
            financial_metrics={},
            stability=10.0,
        )

        assert report is not None
        assert isinstance(report, StrategicReport)

    def test_translator_handles_none_optional_params(
        self,
        translator: SemanticTranslator,
    ) -> None:
        """
        Parámetros opcionales None no causan excepción.
        """
        report = translator.compose_strategic_narrative(
            topological_metrics=TopologicalMetrics(),
            financial_metrics={},
            stability=10.0,
            thermal_metrics=None,
            spectral=None,
            synergy_risk=None,
        )

        assert report is not None

    def test_narrator_handles_none_context(
        self,
        narrator: TelemetryNarrator,
    ) -> None:
        """
        El narrator maneja contexto None graciosamente.
        """
        report = narrator.summarize_execution(None)

        assert report is not None
        assert "verdict" in str(report).lower() or "verdict_code" in report


# =============================================================================
# TEST SUITE 6: SERIALIZACIÓN CRUZADA
# =============================================================================


@pytest.mark.integration
class TestCrossModuleSerialization:
    """
    Verifica que los reportes de ambos módulos son JSON-serializable
    y pueden combinarse en un reporte unificado.
    """

    def test_narrator_report_json_serializable(
        self,
        narrator: TelemetryNarrator,
    ) -> None:
        """Reporte del narrator se serializa/deserializa sin pérdida."""
        context = _create_successful_context()
        report = narrator.summarize_execution(context)

        json_str = json.dumps(report)
        restored = json.loads(json_str)

        assert restored["verdict_code"] == report["verdict_code"]

    def test_translator_report_json_serializable(
        self,
        translator: SemanticTranslator,
    ) -> None:
        """Reporte del translator se serializa sin error."""
        report = translator.compose_strategic_narrative(
            topological_metrics=TopologicalMetrics(),
            financial_metrics={
                "performance": {"recommendation": "ACEPTAR"},
            },
            stability=10.0,
        )

        report_dict = report.to_dict()
        json_str = json.dumps(report_dict)

        assert isinstance(json_str, str)
        assert len(json_str) > 50

    def test_combined_report_json_serializable(
        self,
        narrator: TelemetryNarrator,
        translator: SemanticTranslator,
    ) -> None:
        """
        Reporte combinado (narrator + translator) es JSON-serializable.
        """
        context = _create_successful_context()

        combined = {
            "telemetry": narrator.summarize_execution(context),
            "strategic": translator.compose_strategic_narrative(
                topological_metrics=TopologicalMetrics(),
                financial_metrics={},
                stability=10.0,
            ).to_dict(),
            "timestamp": datetime.utcnow().isoformat(),
        }

        json_str = json.dumps(combined)
        assert isinstance(json_str, str)
        assert len(json_str) > 100


# =============================================================================
# TEST SUITE 7: MATRIZ DE DECISIÓN COMBINADA
# =============================================================================


@pytest.mark.integration
class TestCombinedDecisionMatrix:
    """
    Tests parametrizados de la matriz de decisión combinando
    ambos módulos.

    Variables de entrada:
        - Telemetry status: success, physics_fail, tactics_fail
        - Topología: β₁ (0 = limpia, >0 = ciclos)
        - Financials: ACEPTAR / RECHAZAR
        - Estabilidad: alta (>10) / baja (<1)

    La propiedad verificada es que el veredicto combinado es
    monótonamente peor cuando las condiciones empeoran.
    """

    @pytest.mark.parametrize(
        "telemetry_status, beta_1, recommendation, stability, "
        "expect_narrator_pass, expect_translator_positive",
        [
            # Todo bien
            ("success", 0, "ACEPTAR", 15.0, True, True),
            # Ciclos topológicos
            ("success", 3, "ACEPTAR", 15.0, True, False),
            # Rechazo financiero
            ("success", 0, "RECHAZAR", 15.0, True, False),
            # Baja estabilidad
            ("success", 0, "ACEPTAR", 0.5, True, False),
            # Fallo en PHYSICS
            ("physics_fail", 0, "ACEPTAR", 15.0, False, True),
            # Fallo en TACTICS
            ("tactics_fail", 0, "ACEPTAR", 15.0, False, True),
            # Múltiples problemas
            ("success", 5, "RECHAZAR", 0.5, True, False),
        ],
        ids=[
            "ideal",
            "cyclic-topology",
            "financial-reject",
            "low-stability",
            "physics-fail",
            "tactics-fail",
            "multiple-issues",
        ],
    )
    def test_decision_matrix_entry(
        self,
        narrator: TelemetryNarrator,
        translator: SemanticTranslator,
        telemetry_status: str,
        beta_1: int,
        recommendation: str,
        stability: float,
        expect_narrator_pass: bool,
        expect_translator_positive: bool,
    ) -> None:
        """
        Verifica una entrada de la matriz de decisión.
        """
        # Crear contexto según status
        if telemetry_status == "success":
            context = _create_successful_context()
        elif telemetry_status == "physics_fail":
            context = _create_physics_failure_context()
        elif telemetry_status == "tactics_fail":
            context = _create_tactics_failure_context()
        else:
            context = TelemetryContext()

        # Generar reportes
        telemetry_report = narrator.summarize_execution(context)

        strategic_report = translator.compose_strategic_narrative(
            topological_metrics=TopologicalMetrics(
                beta_0=1, beta_1=beta_1
            ),
            financial_metrics={
                "performance": {
                    "recommendation": recommendation,
                    "profitability_index": 1.0 if recommendation == "ACEPTAR" else 0.7,
                },
            },
            stability=stability,
        )

        # Verificar narrator
        narrator_passed = telemetry_report["verdict_code"] == "APPROVED"
        assert narrator_passed == expect_narrator_pass, (
            f"Narrator: esperado {'APPROVED' if expect_narrator_pass else 'REJECTED'}, "
            f"obtenido {telemetry_report['verdict_code']}."
        )

        # Verificar translator (solo cuando corresponde)
        if expect_translator_positive:
            assert strategic_report.verdict.is_positive, (
                f"Translator: esperado positivo, "
                f"obtenido {strategic_report.verdict.name}."
            )
        else:
            assert strategic_report.verdict.value >= VerdictLevel.REVISAR.value, (
                f"Translator: esperado ≥ REVISAR, "
                f"obtenido {strategic_report.verdict.name}."
            )

    def test_clausura_transitiva_consistency(
        self,
        narrator: TelemetryNarrator,
        translator: SemanticTranslator,
    ) -> None:
        """
        Invariante: Fail(PHYSICS) en telemetría ⟹ translator también
        rechaza (con topología corrupta β₀=0).
        """
        context = _create_physics_failure_context()

        telemetry_report = narrator.summarize_execution(context)
        assert "PHYSICS" in telemetry_report["verdict_code"]

        strategic_report = translator.compose_strategic_narrative(
            topological_metrics=TopologicalMetrics(beta_0=0, beta_1=0),
            financial_metrics={},
            stability=0.1,
        )

        assert strategic_report.verdict == VerdictLevel.RECHAZAR, (
            f"Clausura transitiva violada: PHYSICS falló pero "
            f"translator dio {strategic_report.verdict.name}."
        )


# =============================================================================
# TEST SUITE 8: DETERMINISMO
# =============================================================================


@pytest.mark.integration
class TestDeterminism:
    """
    Verifica que ambos módulos producen resultados determinísticos
    con los mismos inputs.
    """

    def test_narrator_determinism(
        self,
        narrator: TelemetryNarrator,
    ) -> None:
        """Misma telemetría → mismo verdict_code."""
        results = []
        for _ in range(3):
            ctx = _create_successful_context()
            report = narrator.summarize_execution(ctx)
            results.append(report["verdict_code"])

        assert all(r == results[0] for r in results), (
            f"Narrator no determinístico: {results}"
        )

    def test_translator_determinism(
        self,
        translator: SemanticTranslator,
    ) -> None:
        """Mismas métricas → mismo veredicto."""
        results = []
        for _ in range(3):
            report = translator.compose_strategic_narrative(
                topological_metrics=TopologicalMetrics(),
                financial_metrics={
                    "performance": {"recommendation": "ACEPTAR"},
                },
                stability=10.0,
            )
            results.append(report.verdict)

        assert all(r == results[0] for r in results), (
            f"Translator no determinístico: "
            f"{[r.name for r in results]}"
        )


# =============================================================================
# TEST SUITE 9: FACTORY FUNCTIONS
# =============================================================================


class TestFactoryIntegration:
    """Verifica que las factory functions producen instancias correctas."""

    def test_create_both_with_defaults(self) -> None:
        """Factory functions con valores por defecto."""
        narrator = create_narrator()
        translator = create_translator()

        assert isinstance(narrator, TelemetryNarrator)
        assert isinstance(translator, SemanticTranslator)

    def test_create_with_custom_configs(self) -> None:
        """Factory functions con configuraciones personalizadas."""
        narrator = create_narrator(
            step_mapping={"custom_step": Stratum.WISDOM}
        )
        translator = create_translator(
            config=TranslatorConfig(deterministic_market=True)
        )

        assert narrator.step_mapping.get("custom_step") == Stratum.WISDOM
        assert translator.config.deterministic_market is True


# =============================================================================
# TEST SUITE 10: HOMOLOGÍA TOPOLÓGICA
# =============================================================================


@pytest.mark.integration
class TestBettiNumberConsistency:
    """
    Verifica consistencia de los números de Betti entre módulos.

    Para grafos (complejos simpliciales de dimensión 1):
        χ = β₀ - β₁  (Euler-Poincaré)

    Invariantes:
        - β₁ = 0 → narrativa indica estructura limpia
        - β₁ > 0 → narrativa indica ciclos/socavones
        - β₀ = 0 → topología corrupta → rechazo
    """

    @pytest.mark.parametrize(
        "beta_0, beta_1, beta_2, expected_chi",
        [
            (1, 0, 0, 1),    # Árbol conexo (beta2=0)
            (1, 3, 0, -2),   # Conexo con 3 ciclos
            (3, 2, 0, 1),    # 3 componentes, 2 ciclos
            (1, 1, 0, 0),    # Conexo con 1 ciclo
        ],
        ids=["tree", "3-cycles", "3-components", "1-cycle"],
    )
    def test_euler_poincare_consistency(
        self,
        translator: SemanticTranslator,
        beta_0: int,
        beta_1: int,
        beta_2: int,
        expected_chi: int,
    ) -> None:
        """
        χ = β₀ - β₁ + β₂ es consistente para todos los casos.
        """
        calculated_chi = beta_0 - beta_1 + beta_2
        assert calculated_chi == expected_chi, (
            f"χ inconsistente: β₀={beta_0}, β₁={beta_1}, "
            f"χ calculado={calculated_chi}, esperado={expected_chi}."
        )

        topology = TopologicalMetrics(
            beta_0=beta_0,
            beta_1=beta_1,
            beta_2=beta_2,
            euler_characteristic=expected_chi,
        )

        report = translator.compose_strategic_narrative(
            topological_metrics=topology,
            financial_metrics={
                "performance": {"recommendation": "ACEPTAR"},
            },
            stability=10.0,
        )

        if beta_1 > 0:
            _assert_narrative_contains_any(
                report.raw_narrative,
                ["ciclo", "socavón", "agujero", "β₁", "homolog"],
                f"betti-β₁={beta_1}",
            )

    def test_zero_beta_0_indicates_corruption(
        self,
        translator: SemanticTranslator,
    ) -> None:
        """
        β₀ = 0 indica que no hay componentes conexas (datos vacíos
        o corruptos). El translator debe rechazar.
        """
        report = translator.compose_strategic_narrative(
            topological_metrics=TopologicalMetrics(beta_0=0, beta_1=0),
            financial_metrics={
                "performance": {"recommendation": "RECHAZAR"},
            },
            stability=0.1,
        )

        assert report.verdict == VerdictLevel.RECHAZAR, (
            f"β₀=0 debería dar RECHAZAR, "
            f"obtenido {report.verdict.name}."
        )


# =============================================================================
# TEST SUITE 11: FORENSIC EVIDENCE
# =============================================================================


@pytest.mark.integration
class TestForensicEvidenceFormat:
    """
    Verifica que la evidencia forense tiene formato compatible
    entre módulos.
    """

    def test_error_evidence_has_required_fields(
        self,
        narrator: TelemetryNarrator,
    ) -> None:
        """
        Cada item de evidencia forense tiene source, message, type.
        """
        context = _create_physics_failure_context()
        report = narrator.summarize_execution(context)

        evidence = report.get("forensic_evidence", [])

        for item in evidence:
            assert "source" in item, (
                f"Evidencia sin 'source': {item}"
            )
            assert "message" in item, (
                f"Evidencia sin 'message': {item}"
            )
            assert "type" in item, (
                f"Evidencia sin 'type': {item}"
            )
            assert isinstance(item["message"], str), (
                f"message debe ser str: {item['message']}"
            )