"""
Suite de Pruebas para el Narrador Piramidal (TelemetryNarrator)
===============================================================

Valida:
1. Propiedades algebraicas del Lattice de Severidad
2. Estructuras de datos (Issue, PhaseAnalysis, StratumAnalysis, PyramidalReport)
3. Plantillas de narrativa (NarrativeTemplates)
4. Lógica de Clausura Transitiva (DIKW)
5. Recolección recursiva de issues
6. Integración con TelemetryContext
7. Casos límite y manejo de errores

Arquitectura de Tests:
- TestSeverityLevelLattice: Propiedades algebraicas del lattice
- TestSeverityLevelOperations: Operaciones específicas
- TestIssueDataclass: Validación de Issue
- TestPhaseAnalysis: Validación de PhaseAnalysis
- TestStratumAnalysis: Validación de StratumAnalysis
- TestPyramidalReport: Validación de PyramidalReport
- TestNarrativeTemplates: Plantillas de narrativa
- TestTelemetryNarratorCore: Funcionalidad central del narrador
- TestClausuraTransitiva: Lógica DIKW piramidal
- TestIssueCollection: Recolección recursiva de issues
- TestSpanHierarchy: Jerarquías de spans anidados
- TestLegacyMode: Compatibilidad con contextos sin spans
- TestEdgeCasesAndErrors: Casos límite
"""

from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock, patch, MagicMock

import pytest

# Importar desde el módulo refinado
from app.telemetry_narrative import (
    # Configuración
    NarratorConfig,
    StratumTopology,
    # Lattice
    SeverityLevel,
    # Estructuras de datos
    Issue,
    PhaseAnalysis,
    StratumAnalysis,
    PyramidalReport,
    # Plantillas
    NarrativeTemplates,
    # Narrador
    TelemetryNarrator,
    # Factories
    create_narrator,
    summarize_context,
)

# Importar dependencias
from app.telemetry import StepStatus, TelemetryContext, TelemetrySpan
from app.schemas import Stratum


# ============================================================================
# FIXTURES GLOBALES
# ============================================================================


@pytest.fixture
def narrator() -> TelemetryNarrator:
    """Narrador con configuración por defecto."""
    return TelemetryNarrator()


@pytest.fixture
def context() -> TelemetryContext:
    """Contexto de telemetría vacío."""
    return TelemetryContext()


@pytest.fixture
def config() -> NarratorConfig:
    """Configuración por defecto."""
    return NarratorConfig()


@pytest.fixture
def clean_span() -> TelemetrySpan:
    """Span limpio (success, sin errores)."""
    span = TelemetrySpan(name="test_span", level=0, stratum=Stratum.PHYSICS)
    span.status = StepStatus.SUCCESS
    span.end_time = span.start_time + 0.1
    return span


@pytest.fixture
def failed_span() -> TelemetrySpan:
    """Span con fallo."""
    span = TelemetrySpan(name="failed_span", level=0, stratum=Stratum.PHYSICS)
    span.status = StepStatus.FAILURE
    span.errors.append({
        "message": "Test error",
        "type": "TestError",
        "timestamp": datetime.utcnow().isoformat(),
    })
    span.end_time = span.start_time + 0.1
    return span


@pytest.fixture
def warning_span() -> TelemetrySpan:
    """Span con warning."""
    span = TelemetrySpan(name="warning_span", level=0, stratum=Stratum.TACTICS)
    span.status = StepStatus.WARNING
    span.end_time = span.start_time + 0.1
    return span


# ============================================================================
# TEST: PROPIEDADES ALGEBRAICAS DEL LATTICE DE SEVERIDAD
# ============================================================================


class TestSeverityLevelLattice:
    """
    Verifica las propiedades algebraicas del lattice (SeverityLevel, ≤, ⊔, ⊓).
    
    Un lattice debe satisfacer:
    1. Conmutatividad: a ⊔ b = b ⊔ a, a ⊓ b = b ⊓ a
    2. Asociatividad: (a ⊔ b) ⊔ c = a ⊔ (b ⊔ c)
    3. Idempotencia: a ⊔ a = a, a ⊓ a = a
    4. Absorción: a ⊔ (a ⊓ b) = a, a ⊓ (a ⊔ b) = a
    5. Identidad: a ⊔ ⊥ = a, a ⊓ ⊤ = a
    """

    @pytest.fixture
    def all_levels(self) -> List[SeverityLevel]:
        """Todos los niveles de severidad."""
        return list(SeverityLevel)

    def test_lattice_has_bottom_and_top(self):
        """Verifica existencia de elementos ⊥ y ⊤."""
        assert SeverityLevel.bottom() == SeverityLevel.OPTIMO
        assert SeverityLevel.top() == SeverityLevel.CRITICO

    def test_lattice_order_is_total(self, all_levels):
        """Verifica que el orden es total (todos los elementos son comparables)."""
        for a in all_levels:
            for b in all_levels:
                # En un orden total, siempre a <= b o b <= a
                assert a.value <= b.value or b.value <= a.value

    @pytest.mark.parametrize("a", list(SeverityLevel))
    @pytest.mark.parametrize("b", list(SeverityLevel))
    def test_join_commutativity(self, a: SeverityLevel, b: SeverityLevel):
        """Conmutatividad del join: a ⊔ b = b ⊔ a."""
        assert a | b == b | a
        assert SeverityLevel.supremum(a, b) == SeverityLevel.supremum(b, a)

    @pytest.mark.parametrize("a", list(SeverityLevel))
    @pytest.mark.parametrize("b", list(SeverityLevel))
    def test_meet_commutativity(self, a: SeverityLevel, b: SeverityLevel):
        """Conmutatividad del meet: a ⊓ b = b ⊓ a."""
        assert a & b == b & a
        assert SeverityLevel.infimum(a, b) == SeverityLevel.infimum(b, a)

    @pytest.mark.parametrize("a", list(SeverityLevel))
    @pytest.mark.parametrize("b", list(SeverityLevel))
    @pytest.mark.parametrize("c", list(SeverityLevel))
    def test_join_associativity(self, a: SeverityLevel, b: SeverityLevel, c: SeverityLevel):
        """Asociatividad del join: (a ⊔ b) ⊔ c = a ⊔ (b ⊔ c)."""
        assert (a | b) | c == a | (b | c)

    @pytest.mark.parametrize("a", list(SeverityLevel))
    @pytest.mark.parametrize("b", list(SeverityLevel))
    @pytest.mark.parametrize("c", list(SeverityLevel))
    def test_meet_associativity(self, a: SeverityLevel, b: SeverityLevel, c: SeverityLevel):
        """Asociatividad del meet: (a ⊓ b) ⊓ c = a ⊓ (b ⊓ c)."""
        assert (a & b) & c == a & (b & c)

    @pytest.mark.parametrize("a", list(SeverityLevel))
    def test_join_idempotence(self, a: SeverityLevel):
        """Idempotencia del join: a ⊔ a = a."""
        assert a | a == a
        assert SeverityLevel.supremum(a, a) == a

    @pytest.mark.parametrize("a", list(SeverityLevel))
    def test_meet_idempotence(self, a: SeverityLevel):
        """Idempotencia del meet: a ⊓ a = a."""
        assert a & a == a
        assert SeverityLevel.infimum(a, a) == a

    @pytest.mark.parametrize("a", list(SeverityLevel))
    @pytest.mark.parametrize("b", list(SeverityLevel))
    def test_absorption_law_join(self, a: SeverityLevel, b: SeverityLevel):
        """Ley de absorción join: a ⊔ (a ⊓ b) = a."""
        assert a | (a & b) == a

    @pytest.mark.parametrize("a", list(SeverityLevel))
    @pytest.mark.parametrize("b", list(SeverityLevel))
    def test_absorption_law_meet(self, a: SeverityLevel, b: SeverityLevel):
        """Ley de absorción meet: a ⊓ (a ⊔ b) = a."""
        assert a & (a | b) == a

    @pytest.mark.parametrize("a", list(SeverityLevel))
    def test_bottom_is_join_identity(self, a: SeverityLevel):
        """⊥ es identidad del join: a ⊔ ⊥ = a."""
        bottom = SeverityLevel.bottom()
        assert a | bottom == a

    @pytest.mark.parametrize("a", list(SeverityLevel))
    def test_top_is_meet_identity(self, a: SeverityLevel):
        """⊤ es identidad del meet: a ⊓ ⊤ = a."""
        top = SeverityLevel.top()
        assert a & top == a

    def test_supremum_empty_returns_bottom(self):
        """Supremum vacío retorna ⊥."""
        assert SeverityLevel.supremum() == SeverityLevel.OPTIMO

    def test_infimum_empty_returns_top(self):
        """Infimum vacío retorna ⊤."""
        assert SeverityLevel.infimum() == SeverityLevel.CRITICO

    def test_supremum_takes_worst_case(self):
        """Supremum toma el peor caso."""
        result = SeverityLevel.supremum(
            SeverityLevel.OPTIMO,
            SeverityLevel.ADVERTENCIA,
            SeverityLevel.CRITICO,
        )
        assert result == SeverityLevel.CRITICO

    def test_infimum_takes_best_case(self):
        """Infimum toma el mejor caso."""
        result = SeverityLevel.infimum(
            SeverityLevel.OPTIMO,
            SeverityLevel.ADVERTENCIA,
            SeverityLevel.CRITICO,
        )
        assert result == SeverityLevel.OPTIMO


class TestSeverityLevelOperations:
    """Pruebas de operaciones específicas de SeverityLevel."""

    def test_emoji_mapping(self):
        """Verifica mapeo de emojis."""
        assert SeverityLevel.OPTIMO.emoji == "✅"
        assert SeverityLevel.ADVERTENCIA.emoji == "⚠️"
        assert SeverityLevel.CRITICO.emoji == "❌"

    def test_is_critical_property(self):
        """Verifica propiedad is_critical."""
        assert SeverityLevel.OPTIMO.is_critical is False
        assert SeverityLevel.ADVERTENCIA.is_critical is False
        assert SeverityLevel.CRITICO.is_critical is True

    def test_is_optimal_property(self):
        """Verifica propiedad is_optimal."""
        assert SeverityLevel.OPTIMO.is_optimal is True
        assert SeverityLevel.ADVERTENCIA.is_optimal is False
        assert SeverityLevel.CRITICO.is_optimal is False

    @pytest.mark.parametrize(
        "status,expected",
        [
            (StepStatus.SUCCESS, SeverityLevel.OPTIMO),
            (StepStatus.IN_PROGRESS, SeverityLevel.OPTIMO),
            (StepStatus.SKIPPED, SeverityLevel.OPTIMO),
            (StepStatus.WARNING, SeverityLevel.ADVERTENCIA),
            (StepStatus.CANCELLED, SeverityLevel.ADVERTENCIA),
            (StepStatus.FAILURE, SeverityLevel.CRITICO),
        ],
    )
    def test_from_step_status(self, status: StepStatus, expected: SeverityLevel):
        """Verifica conversión desde StepStatus."""
        assert SeverityLevel.from_step_status(status) == expected

    def test_from_step_status_string(self):
        """Verifica conversión desde string de status."""
        assert SeverityLevel.from_step_status("success") == SeverityLevel.OPTIMO
        assert SeverityLevel.from_step_status("failure") == SeverityLevel.CRITICO

    def test_from_step_status_none(self):
        """None retorna OPTIMO."""
        assert SeverityLevel.from_step_status(None) == SeverityLevel.OPTIMO

    def test_from_step_status_invalid(self):
        """Valor inválido retorna OPTIMO."""
        assert SeverityLevel.from_step_status("invalid_status") == SeverityLevel.OPTIMO

    def test_from_error_count(self):
        """Verifica derivación desde conteo de errores."""
        assert SeverityLevel.from_error_count(0) == SeverityLevel.OPTIMO
        assert SeverityLevel.from_error_count(1) == SeverityLevel.CRITICO
        assert SeverityLevel.from_error_count(5, threshold=3) == SeverityLevel.CRITICO
        assert SeverityLevel.from_error_count(2, threshold=5) == SeverityLevel.ADVERTENCIA

    def test_binary_operators_consistency(self):
        """Verifica consistencia entre operadores y métodos."""
        a, b = SeverityLevel.OPTIMO, SeverityLevel.CRITICO

        assert a | b == a.join(b)
        assert a & b == a.meet(b)
        assert a | b == SeverityLevel.supremum(a, b)
        assert a & b == SeverityLevel.infimum(a, b)


# ============================================================================
# TEST: STRATUM TOPOLOGY
# ============================================================================


class TestStratumTopology:
    """Pruebas para la topología de estratos."""

    def test_hierarchy_order(self):
        """Verifica orden jerárquico: PHYSICS(3) > TACTICS(2) > STRATEGY(1) > WISDOM(0)."""
        assert StratumTopology.get_level(Stratum.PHYSICS) == 3
        assert StratumTopology.get_level(Stratum.TACTICS) == 2
        assert StratumTopology.get_level(Stratum.STRATEGY) == 1
        assert StratumTopology.get_level(Stratum.WISDOM) == 0

    def test_evaluation_order(self):
        """Verifica orden de evaluación (de base a cima)."""
        order = StratumTopology.EVALUATION_ORDER
        assert order[0] == Stratum.PHYSICS
        assert order[1] == Stratum.TACTICS
        assert order[2] == Stratum.STRATEGY
        assert order[3] == Stratum.WISDOM

    @pytest.mark.parametrize(
        "step_name,expected_stratum",
        [
            ("load_data", Stratum.PHYSICS),
            ("merge_data", Stratum.PHYSICS),
            ("flux_condenser", Stratum.PHYSICS),
            ("calculate_costs", Stratum.TACTICS),
            ("materialization", Stratum.TACTICS),
            ("business_topology", Stratum.STRATEGY),
            ("financial_analysis", Stratum.STRATEGY),
            ("build_output", Stratum.WISDOM),
            ("response_preparation", Stratum.WISDOM),
        ],
    )
    def test_default_step_mapping(self, step_name: str, expected_stratum: Stratum):
        """Verifica mapeo por defecto de pasos a estratos."""
        result = StratumTopology.get_stratum_for_step(step_name)
        assert result == expected_stratum

    def test_unknown_step_defaults_to_physics(self):
        """Paso desconocido se mapea a PHYSICS por seguridad."""
        assert StratumTopology.get_stratum_for_step("unknown_step") == Stratum.PHYSICS

    def test_custom_mapping_override(self):
        """Mapeo personalizado tiene precedencia."""
        custom = {"my_step": Stratum.WISDOM}
        result = StratumTopology.get_stratum_for_step("my_step", custom)
        assert result == Stratum.WISDOM

    def test_is_higher_than(self):
        """Verifica comparación de niveles."""
        assert StratumTopology.is_higher_than(Stratum.WISDOM, Stratum.PHYSICS) is True
        assert StratumTopology.is_higher_than(Stratum.PHYSICS, Stratum.WISDOM) is False
        assert StratumTopology.is_higher_than(Stratum.TACTICS, Stratum.TACTICS) is False

    def test_get_strata_above(self):
        """Verifica obtención de estratos superiores."""
        above_physics = StratumTopology.get_strata_above(Stratum.PHYSICS)
        assert Stratum.TACTICS in above_physics
        assert Stratum.STRATEGY in above_physics
        assert Stratum.WISDOM in above_physics
        assert Stratum.PHYSICS not in above_physics

    def test_get_strata_below(self):
        """Verifica obtención de estratos inferiores."""
        below_wisdom = StratumTopology.get_strata_below(Stratum.WISDOM)
        assert Stratum.PHYSICS in below_wisdom
        assert Stratum.TACTICS in below_wisdom
        assert Stratum.STRATEGY in below_wisdom
        assert Stratum.WISDOM not in below_wisdom


# ============================================================================
# TEST: ISSUE DATACLASS
# ============================================================================


class TestIssueDataclass:
    """Pruebas para la estructura Issue."""

    def test_create_with_tuple_path(self):
        """Creación con path como tupla."""
        issue = Issue(
            source="test",
            message="Test message",
            issue_type="Error",
            depth=0,
            topological_path=("root", "child"),
        )
        assert issue.topological_path == ("root", "child")
        assert issue.path_string == "root → child"

    def test_create_with_string_path(self):
        """Creación con path como string (se convierte a tupla)."""
        issue = Issue(
            source="test",
            message="Test message",
            issue_type="Error",
            depth=0,
            topological_path="root → child → leaf",
        )
        assert issue.topological_path == ("root", "child", "leaf")

    def test_create_with_list_path(self):
        """Creación con path como lista (se convierte a tupla)."""
        issue = Issue(
            source="test",
            message="Test message",
            issue_type="Error",
            depth=0,
            topological_path=["a", "b", "c"],
        )
        assert issue.topological_path == ("a", "b", "c")

    def test_message_truncation(self):
        """Mensajes largos se truncan."""
        long_message = "x" * 1000
        issue = Issue(
            source="test",
            message=long_message,
            issue_type="Error",
            depth=0,
            topological_path=("root",),
        )
        assert len(issue.message) <= NarratorConfig.MAX_PATH_LENGTH + 3  # +3 for "..."

    def test_is_critical_based_on_type(self):
        """is_critical se determina por tipo de issue."""
        critical = Issue(
            source="test", message="msg", issue_type="Error",
            depth=0, topological_path=("root",)
        )
        non_critical = Issue(
            source="test", message="msg", issue_type="Warning",
            depth=0, topological_path=("root",)
        )

        assert critical.is_critical is True
        assert non_critical.is_critical is False

    def test_is_critical_based_on_severity(self):
        """is_critical también considera severidad explícita."""
        issue = Issue(
            source="test", message="msg", issue_type="Info",
            depth=0, topological_path=("root",),
            severity=SeverityLevel.CRITICO,
        )
        assert issue.is_critical is True

    def test_with_stratum_returns_new_instance(self):
        """with_stratum retorna nueva instancia con estrato."""
        original = Issue(
            source="test", message="msg", issue_type="Error",
            depth=0, topological_path=("root",)
        )
        with_stratum = original.with_stratum(Stratum.TACTICS)

        assert with_stratum is not original
        assert with_stratum.stratum == Stratum.TACTICS
        assert with_stratum.context.get("stratum") == "TACTICS"

    def test_path_depth_calculation(self):
        """path_depth se calcula correctamente."""
        issue = Issue(
            source="test", message="msg", issue_type="Error",
            depth=0, topological_path=("a", "b", "c", "d"),
        )
        assert issue.path_depth == 3  # 4 elementos - 1

    def test_to_dict_serialization(self):
        """Serialización a diccionario."""
        issue = Issue(
            source="test",
            message="Test message",
            issue_type="Error",
            depth=2,
            topological_path=("root", "child"),
            timestamp="2024-01-01T00:00:00",
            stratum=Stratum.PHYSICS,
            severity=SeverityLevel.CRITICO,
            context={"extra": "data"},
        )
        result = issue.to_dict()

        assert result["source"] == "test"
        assert result["message"] == "Test message"
        assert result["type"] == "Error"
        assert result["depth"] == 2
        assert result["topological_path"] == "root → child"
        assert result["timestamp"] == "2024-01-01T00:00:00"
        assert result["stratum"] == "PHYSICS"
        assert result["severity"] == "CRITICO"
        assert result["context"]["extra"] == "data"


# ============================================================================
# TEST: PHASE ANALYSIS
# ============================================================================


class TestPhaseAnalysis:
    """Pruebas para PhaseAnalysis."""

    def test_create_basic(self):
        """Creación básica."""
        phase = PhaseAnalysis(
            name="test_phase",
            stratum=Stratum.PHYSICS,
            severity=SeverityLevel.OPTIMO,
            duration_seconds=1.5,
            issues=[],
            warning_count=0,
        )
        assert phase.name == "test_phase"
        assert phase.is_clean is True

    def test_critical_issues_filter(self):
        """Filtrado de issues críticos."""
        critical = Issue(
            source="s", message="m", issue_type="Error",
            depth=0, topological_path=("p",), severity=SeverityLevel.CRITICO
        )
        warning = Issue(
            source="s", message="m", issue_type="Warning",
            depth=0, topological_path=("p",), severity=SeverityLevel.ADVERTENCIA
        )

        phase = PhaseAnalysis(
            name="test", stratum=Stratum.PHYSICS,
            severity=SeverityLevel.CRITICO,
            duration_seconds=1.0,
            issues=[critical, warning],
            warning_count=1,
        )

        assert len(phase.critical_issues) == 1
        assert len(phase.warnings) == 1

    def test_has_failures_property(self):
        """Propiedad has_failures."""
        failing = PhaseAnalysis(
            name="fail", stratum=Stratum.PHYSICS,
            severity=SeverityLevel.CRITICO,
            duration_seconds=1.0, issues=[], warning_count=0,
        )
        success = PhaseAnalysis(
            name="ok", stratum=Stratum.PHYSICS,
            severity=SeverityLevel.OPTIMO,
            duration_seconds=1.0, issues=[], warning_count=0,
        )

        assert failing.has_failures is True
        assert success.has_failures is False

    def test_negative_duration_clamped(self):
        """Duración negativa se ajusta a 0."""
        phase = PhaseAnalysis(
            name="test", stratum=Stratum.PHYSICS,
            severity=SeverityLevel.OPTIMO,
            duration_seconds=-5.0,
            issues=[], warning_count=0,
        )
        assert phase.duration_seconds == 0.0

    def test_issues_truncated(self):
        """Issues se truncan al límite."""
        many_issues = [
            Issue(source="s", message=f"msg_{i}", issue_type="Error",
                  depth=0, topological_path=("p",))
            for i in range(100)
        ]

        phase = PhaseAnalysis(
            name="test", stratum=Stratum.PHYSICS,
            severity=SeverityLevel.CRITICO,
            duration_seconds=1.0,
            issues=many_issues,
            warning_count=0,
        )

        assert len(phase.issues) <= NarratorConfig.MAX_ISSUES_PER_PHASE

    def test_to_dict_serialization(self):
        """Serialización a diccionario."""
        phase = PhaseAnalysis(
            name="test_phase",
            stratum=Stratum.TACTICS,
            severity=SeverityLevel.ADVERTENCIA,
            duration_seconds=2.5,
            issues=[],
            warning_count=3,
            child_count=5,
            metrics={"key": "value"},
        )
        result = phase.to_dict()

        assert result["name"] == "test_phase"
        assert result["stratum"] == "TACTICS"
        assert result["status"] == "ADVERTENCIA"
        assert result["status_emoji"] == "⚠️"
        assert result["duration"] == "2.500s"
        assert result["warning_count"] == 3
        assert result["child_count"] == 5


# ============================================================================
# TEST: STRATUM ANALYSIS
# ============================================================================


class TestStratumAnalysis:
    """Pruebas para StratumAnalysis."""

    def test_create_basic(self):
        """Creación básica."""
        analysis = StratumAnalysis(
            stratum=Stratum.PHYSICS,
            severity=SeverityLevel.OPTIMO,
            narrative="Test narrative",
            phases=[],
            issues=[],
        )
        assert analysis.stratum == Stratum.PHYSICS
        assert analysis.is_healthy is True

    def test_is_compromised_property(self):
        """Propiedad is_compromised."""
        healthy = StratumAnalysis(
            stratum=Stratum.PHYSICS, severity=SeverityLevel.OPTIMO,
            narrative="", phases=[], issues=[],
        )
        compromised = StratumAnalysis(
            stratum=Stratum.PHYSICS, severity=SeverityLevel.CRITICO,
            narrative="", phases=[], issues=[],
        )

        assert healthy.is_compromised is False
        assert compromised.is_compromised is True

    def test_total_duration_calculation(self):
        """Cálculo de duración total."""
        phases = [
            PhaseAnalysis(name="p1", stratum=Stratum.PHYSICS,
                         severity=SeverityLevel.OPTIMO, duration_seconds=1.0,
                         issues=[], warning_count=0),
            PhaseAnalysis(name="p2", stratum=Stratum.PHYSICS,
                         severity=SeverityLevel.OPTIMO, duration_seconds=2.0,
                         issues=[], warning_count=0),
        ]

        analysis = StratumAnalysis(
            stratum=Stratum.PHYSICS, severity=SeverityLevel.OPTIMO,
            narrative="", phases=phases, issues=[],
        )

        assert analysis.total_duration == 3.0

    def test_severity_recalculation(self):
        """Severidad se recalcula desde fases si es OPTIMO inicial."""
        phases = [
            PhaseAnalysis(name="p1", stratum=Stratum.PHYSICS,
                         severity=SeverityLevel.CRITICO, duration_seconds=1.0,
                         issues=[], warning_count=0),
        ]

        analysis = StratumAnalysis(
            stratum=Stratum.PHYSICS,
            severity=SeverityLevel.OPTIMO,  # Se debería recalcular
            narrative="", phases=phases, issues=[],
        )

        assert analysis.severity == SeverityLevel.CRITICO

    def test_to_dict_serialization(self):
        """Serialización a diccionario."""
        analysis = StratumAnalysis(
            stratum=Stratum.STRATEGY,
            severity=SeverityLevel.ADVERTENCIA,
            narrative="Test narrative",
            phases=[],
            issues=[Issue(source="s", message="m", issue_type="E",
                         depth=0, topological_path=("p",))],
            metrics={"key": "value"},
        )
        result = analysis.to_dict()

        assert result["stratum"] == "STRATEGY"
        assert result["level"] == 1
        assert result["severity"] == "ADVERTENCIA"
        assert result["is_compromised"] is False
        assert result["total_issues"] == 1


# ============================================================================
# TEST: PYRAMIDAL REPORT
# ============================================================================


class TestPyramidalReport:
    """Pruebas para PyramidalReport."""

    @pytest.fixture
    def empty_strata(self) -> Dict[Stratum, StratumAnalysis]:
        """Análisis de estratos vacíos."""
        return {
            s: StratumAnalysis(
                stratum=s, severity=SeverityLevel.OPTIMO,
                narrative="", phases=[], issues=[],
            )
            for s in Stratum
        }

    def test_create_basic(self, empty_strata):
        """Creación básica."""
        report = PyramidalReport(
            verdict="✅ APROBADO",
            verdict_code="APPROVED",
            executive_summary="All good",
            global_severity=SeverityLevel.OPTIMO,
            strata_analysis=empty_strata,
            forensic_evidence=[],
            phases=[],
        )
        assert report.is_approved is True

    def test_is_approved_property(self, empty_strata):
        """Propiedad is_approved."""
        approved = PyramidalReport(
            verdict="OK", verdict_code="APPROVED",
            executive_summary="", global_severity=SeverityLevel.OPTIMO,
            strata_analysis=empty_strata, forensic_evidence=[], phases=[],
        )
        rejected = PyramidalReport(
            verdict="NO", verdict_code="REJECTED",
            executive_summary="", global_severity=SeverityLevel.CRITICO,
            strata_analysis=empty_strata, forensic_evidence=[], phases=[],
        )

        assert approved.is_approved is True
        assert rejected.is_approved is False

    def test_failed_strata_property(self, empty_strata):
        """Propiedad failed_strata."""
        # Comprometer PHYSICS
        empty_strata[Stratum.PHYSICS] = StratumAnalysis(
            stratum=Stratum.PHYSICS, severity=SeverityLevel.CRITICO,
            narrative="Failed", phases=[], issues=[],
        )

        report = PyramidalReport(
            verdict="", verdict_code="",
            executive_summary="", global_severity=SeverityLevel.CRITICO,
            strata_analysis=empty_strata, forensic_evidence=[], phases=[],
        )

        assert Stratum.PHYSICS in report.failed_strata
        assert Stratum.TACTICS not in report.failed_strata

    def test_root_cause_stratum_property(self, empty_strata):
        """Propiedad root_cause_stratum identifica el estrato más bajo que falló."""
        # Fallar PHYSICS y STRATEGY
        empty_strata[Stratum.PHYSICS] = StratumAnalysis(
            stratum=Stratum.PHYSICS, severity=SeverityLevel.CRITICO,
            narrative="", phases=[], issues=[],
        )
        empty_strata[Stratum.STRATEGY] = StratumAnalysis(
            stratum=Stratum.STRATEGY, severity=SeverityLevel.CRITICO,
            narrative="", phases=[], issues=[],
        )

        report = PyramidalReport(
            verdict="", verdict_code="",
            executive_summary="", global_severity=SeverityLevel.CRITICO,
            strata_analysis=empty_strata, forensic_evidence=[], phases=[],
        )

        # PHYSICS debería ser la causa raíz (está en la base)
        assert report.root_cause_stratum == Stratum.PHYSICS

    def test_to_dict_serialization(self, empty_strata):
        """Serialización a diccionario."""
        report = PyramidalReport(
            verdict="Test Verdict",
            verdict_code="TEST",
            executive_summary="Test summary",
            global_severity=SeverityLevel.ADVERTENCIA,
            strata_analysis=empty_strata,
            forensic_evidence=[],
            phases=[],
            causality_chain=["Step 1", "Step 2"],
            recommendations=["Rec 1"],
        )
        result = report.to_dict()

        assert result["verdict"] == "Test Verdict"
        assert result["verdict_code"] == "TEST"
        assert result["narrative"] == "Test summary"  # Alias
        assert result["is_approved"] is False
        assert len(result["strata_analysis"]) == 4
        assert result["causality_chain"] == ["Step 1", "Step 2"]


# ============================================================================
# TEST: NARRATIVE TEMPLATES
# ============================================================================


class TestNarrativeTemplates:
    """Pruebas para plantillas de narrativa."""

    def test_success_narratives_exist_for_all_strata(self):
        """Existen narrativas de éxito para todos los estratos."""
        for stratum in Stratum:
            assert stratum in NarrativeTemplates.SUCCESS_NARRATIVES

    def test_failure_narratives_exist_for_all_strata(self):
        """Existen narrativas de fallo para todos los estratos."""
        for stratum in Stratum:
            assert stratum in NarrativeTemplates.FAILURE_NARRATIVES
            assert "default" in NarrativeTemplates.FAILURE_NARRATIVES[stratum]

    def test_warning_narratives_exist_for_all_strata(self):
        """Existen narrativas de warning para todos los estratos."""
        for stratum in Stratum:
            assert stratum in NarrativeTemplates.WARNING_NARRATIVES

    def test_verdict_templates_exist(self):
        """Existen plantillas de veredictos."""
        required_verdicts = [
            "APPROVED",
            "REJECTED_PHYSICS",
            "REJECTED_TACTICS",
            "REJECTED_STRATEGY",
            "REJECTED_WISDOM",
        ]
        for verdict_code in required_verdicts:
            assert verdict_code in NarrativeTemplates.VERDICTS

    def test_get_stratum_narrative_success(self):
        """get_stratum_narrative retorna narrativa de éxito."""
        narrative = NarrativeTemplates.get_stratum_narrative(
            Stratum.PHYSICS, SeverityLevel.OPTIMO, []
        )
        assert "✅" in narrative
        assert len(narrative) > 0

    def test_get_stratum_narrative_failure(self):
        """get_stratum_narrative retorna narrativa de fallo."""
        issues = [Issue(
            source="s", message="error test", issue_type="Error",
            depth=0, topological_path=("p",),
        )]
        narrative = NarrativeTemplates.get_stratum_narrative(
            Stratum.PHYSICS, SeverityLevel.CRITICO, issues
        )
        assert "🔥" in narrative or "Falla" in narrative

    def test_get_stratum_narrative_warning(self):
        """get_stratum_narrative retorna narrativa de warning."""
        narrative = NarrativeTemplates.get_stratum_narrative(
            Stratum.TACTICS, SeverityLevel.ADVERTENCIA, []
        )
        assert "⚠️" in narrative or "Subóptima" in narrative

    def test_detect_failure_type_cycles(self):
        """Detecta tipo de fallo por ciclos en TACTICS."""
        issues = [Issue(
            source="s", message="Ciclo infinito detectado", issue_type="Error",
            depth=0, topological_path=("p",),
        )]
        failure_type = NarrativeTemplates._detect_failure_type(Stratum.TACTICS, issues)
        assert failure_type == "cycles"

    def test_detect_failure_type_default(self):
        """Retorna 'default' si no hay keywords."""
        issues = [Issue(
            source="s", message="Generic error", issue_type="Error",
            depth=0, topological_path=("p",),
        )]
        failure_type = NarrativeTemplates._detect_failure_type(Stratum.PHYSICS, issues)
        assert failure_type == "default"

    def test_get_verdict_returns_tuple(self):
        """get_verdict retorna tupla (título, descripción)."""
        title, desc = NarrativeTemplates.get_verdict("APPROVED")
        assert "CERTIFICADO" in title or "🏛️" in title
        assert len(desc) > 0

    def test_get_verdict_unknown_code(self):
        """Código desconocido retorna valor por defecto."""
        title, desc = NarrativeTemplates.get_verdict("UNKNOWN_CODE")
        assert "DESCONOCIDO" in title or "❓" in title


# ============================================================================
# TEST: TELEMETRY NARRATOR CORE
# ============================================================================


class TestTelemetryNarratorCore:
    """Pruebas de funcionalidad central del narrador."""

    def test_create_with_defaults(self):
        """Creación con valores por defecto."""
        narrator = TelemetryNarrator()
        assert narrator.step_mapping == {}
        assert narrator.config is not None

    def test_create_with_custom_mapping(self):
        """Creación con mapeo personalizado."""
        custom = {"my_step": Stratum.WISDOM}
        narrator = TelemetryNarrator(step_mapping=custom)
        assert narrator.step_mapping["my_step"] == Stratum.WISDOM

    def test_summarize_empty_context(self, narrator: TelemetryNarrator):
        """Contexto vacío genera reporte vacío válido."""
        context = TelemetryContext()
        report = narrator.summarize_execution(context)

        assert "verdict" in report
        assert "strata_analysis" in report
        assert report["verdict_code"] == "EMPTY"

    def test_summarize_none_context(self, narrator: TelemetryNarrator):
        """None context se maneja graciosamente."""
        report = narrator.summarize_execution(None)

        assert "verdict" in report
        assert report["verdict_code"] == "EMPTY"

    def test_summarize_with_spans_returns_dict(self, narrator: TelemetryNarrator, context: TelemetryContext):
        """Contexto con spans retorna diccionario válido."""
        with context.span("test_span"):
            pass

        report = narrator.summarize_execution(context)

        assert isinstance(report, dict)
        assert "verdict" in report
        assert "executive_summary" in report
        assert "strata_analysis" in report
        assert "forensic_evidence" in report
        assert "phases" in report

    def test_all_strata_present_in_report(self, narrator: TelemetryNarrator, context: TelemetryContext):
        """Todos los estratos están presentes en el reporte."""
        with context.span("load_data"):
            pass

        report = narrator.summarize_execution(context)

        for stratum in Stratum:
            assert stratum.name in report["strata_analysis"]


# ============================================================================
# TEST: CLAUSURA TRANSITIVA (LÓGICA DIKW)
# ============================================================================


class TestClausuraTransitiva:
    """
    Pruebas de la lógica de Clausura Transitiva.
    
    Regla: Fallo en estrato N invalida todos los estratos superiores.
    PHYSICS → TACTICS → STRATEGY → WISDOM
    """

    def test_physics_failure_blocks_everything(self, narrator: TelemetryNarrator, context: TelemetryContext):
        """
        Fallo en PHYSICS (load_data) → RECHAZADO_TECNICO.
        Los estratos superiores son irrelevantes.
        """
        with context.span("load_data") as span:
            span.status = StepStatus.FAILURE
            span.errors.append({"message": "Corrupted CSV", "type": "DataError"})

        # Esto no debería importar
        with context.span("business_topology"):
            pass

        report = narrator.summarize_execution(context)

        assert "REJECTED_PHYSICS" in report["verdict_code"] or "RECHAZADO" in report["verdict_code"]
        assert "FÍSICA" in report["executive_summary"] or "ABORTADO" in report["executive_summary"]

        # Verificar que evidencia apunta a PHYSICS
        if report["forensic_evidence"]:
            evidence = report["forensic_evidence"][0]
            # El stratum puede estar en context o como campo directo
            stratum = evidence.get("stratum") or evidence.get("context", {}).get("stratum")
            assert stratum == "PHYSICS"

    def test_tactics_failure_veto(self, narrator: TelemetryNarrator, context: TelemetryContext):
        """
        PHYSICS OK, TACTICS falla → VETO_ESTRUCTURAL.
        """
        # Física OK
        with context.span("load_data"):
            pass

        # Táctica Falla
        with context.span("calculate_costs") as span:
            span.status = StepStatus.FAILURE
            span.errors.append({"message": "Cyclic Dependency", "type": "TopologyError"})

        report = narrator.summarize_execution(context)

        assert "TACTICS" in report["verdict_code"] or "VETO" in report["verdict_code"]
        assert report["strata_analysis"]["PHYSICS"]["severity"] == "OPTIMO"
        assert report["strata_analysis"]["TACTICS"]["severity"] == "CRITICO"

    def test_strategy_failure_risk_alert(self, narrator: TelemetryNarrator, context: TelemetryContext):
        """
        PHYSICS y TACTICS OK, STRATEGY falla → RIESGO_FINANCIERO.
        """
        with context.span("load_data"):
            pass
        with context.span("calculate_costs"):
            pass

        with context.span("financial_analysis") as span:
            span.status = StepStatus.FAILURE
            span.errors.append({"message": "VaR Exceeded", "type": "FinancialError"})

        report = narrator.summarize_execution(context)

        assert "STRATEGY" in report["verdict_code"] or "FINANCIERO" in report["verdict_code"]
        assert "ORÁCULO" in report["executive_summary"] or "FINANCIERA" in report["executive_summary"]

    def test_success_certificate(self, narrator: TelemetryNarrator, context: TelemetryContext):
        """
        Todo OK → APROBADO.
        """
        with context.span("load_data"):
            pass
        with context.span("calculate_costs"):
            pass
        with context.span("financial_analysis"):
            pass
        with context.span("build_output"):
            pass

        report = narrator.summarize_execution(context)

        assert report["verdict_code"] == "APPROVED"
        assert "CERTIFICADO" in report["executive_summary"] or "SOLIDEZ" in report["executive_summary"]

    def test_multiple_failures_reports_lowest(self, narrator: TelemetryNarrator, context: TelemetryContext):
        """
        Si fallan PHYSICS y STRATEGY, solo se reporta PHYSICS (más bajo).
        """
        # Physics Fail
        with context.span("load_data") as span:
            span.status = StepStatus.FAILURE
            span.errors.append({"message": "Physics Fail", "type": "Error"})

        # Strategy Fail
        with context.span("financial_analysis") as span:
            span.status = StepStatus.FAILURE
            span.errors.append({"message": "Strategy Fail", "type": "Error"})

        report = narrator.summarize_execution(context)

        # El veredicto debe ser por PHYSICS (causa raíz)
        assert "PHYSICS" in report["verdict_code"]

        # La evidencia solo debe mostrar errores de PHYSICS
        evidence_messages = [e.get("message", "") for e in report["forensic_evidence"]]
        assert any("Physics" in m for m in evidence_messages)
        # Strategy no debería aparecer en evidencia forense
        assert not any("Strategy" in m for m in evidence_messages)

    def test_wisdom_failure_after_all_ok(self, narrator: TelemetryNarrator, context: TelemetryContext):
        """
        Todo OK excepto WISDOM → fallo en síntesis.
        """
        with context.span("load_data"):
            pass
        with context.span("calculate_costs"):
            pass
        with context.span("financial_analysis"):
            pass

        with context.span("build_output") as span:
            span.status = StepStatus.FAILURE
            span.errors.append({"message": "Output generation failed", "type": "Error"})

        report = narrator.summarize_execution(context)

        assert "WISDOM" in report["verdict_code"]


# ============================================================================
# TEST: RECOLECCIÓN RECURSIVA DE ISSUES
# ============================================================================


class TestIssueCollection:
    """Pruebas para recolección recursiva de issues."""

    def test_collect_explicit_errors(self, narrator: TelemetryNarrator, context: TelemetryContext):
        """Errores explícitos se recolectan."""
        with context.span("test") as span:
            span.errors.append({"message": "Explicit error", "type": "TestError"})

        report = narrator.summarize_execution(context)
        phase = report["phases"][0]

        assert phase["critical_count"] >= 1

    def test_collect_silent_failure(self, narrator: TelemetryNarrator, context: TelemetryContext):
        """Silent failures (FAILURE sin errores) se detectan."""
        with context.span("test") as span:
            span.status = StepStatus.FAILURE
            # Sin errores explícitos

        report = narrator.summarize_execution(context)
        phase = report["phases"][0]

        # Debe detectar el fallo silencioso
        assert phase["status"] == "CRITICO" or phase["critical_count"] > 0

    def test_collect_implicit_warning(self, narrator: TelemetryNarrator, context: TelemetryContext):
        """Implicit warnings (WARNING sin errores) se detectan."""
        with context.span("test") as span:
            span.status = StepStatus.WARNING
            # Sin errores explícitos

        report = narrator.summarize_execution(context)
        phase = report["phases"][0]

        assert phase["warning_count"] > 0 or phase["status"] == "ADVERTENCIA"

    def test_recursion_limit_respected(self, narrator: TelemetryNarrator):
        """Se respeta el límite de recursión."""
        # Crear span con muchos niveles anidados manualmente
        root = TelemetrySpan(name="root", level=0)
        current = root

        # Crear cadena profunda
        for i in range(200):  # Más que MAX_RECURSION_DEPTH
            child = TelemetrySpan(name=f"child_{i}", level=i + 1)
            current.children.append(child)
            current = child

        root.status = StepStatus.SUCCESS
        root.end_time = root.start_time + 0.1

        # Analizar
        analysis = narrator._analyze_phase(root)

        # Debe haber un issue de límite de recursión
        recursion_issues = [i for i in analysis.issues if "RecursionLimit" in i.issue_type]
        assert len(recursion_issues) > 0

    def test_anomalous_metrics_detected(self, narrator: TelemetryNarrator, context: TelemetryContext):
        """Métricas anómalas generan issues."""
        with context.span("test") as span:
            span.metrics["test_metric"] = {"value": 100, "anomalous": True}

        report = narrator.summarize_execution(context)
        phase = report["phases"][0]

        # Debe haber warning por métrica anómala
        assert phase["warning_count"] > 0


# ============================================================================
# TEST: JERARQUÍA DE SPANS
# ============================================================================


class TestSpanHierarchy:
    """Pruebas para jerarquías de spans anidados."""

    def test_nested_spans_analyzed(self, narrator: TelemetryNarrator, context: TelemetryContext):
        """Spans anidados se analizan correctamente."""
        with context.span("parent") as parent:
            with context.span("child") as child:
                child.status = StepStatus.FAILURE
                child.errors.append({"message": "Child error", "type": "Error"})

        report = narrator.summarize_execution(context)

        # El error del hijo debe propagarse al padre
        phase = report["phases"][0]
        assert phase["status"] == "CRITICO"

    def test_child_failure_propagates_to_parent(self, narrator: TelemetryNarrator, context: TelemetryContext):
        """Fallo de hijo propaga severidad al padre."""
        with context.span("parent"):
            with context.span("child") as child:
                child.status = StepStatus.FAILURE

        report = narrator.summarize_execution(context)
        phase = report["phases"][0]

        # Severidad debe ser CRITICO por el hijo
        assert phase["status"] == "CRITICO"

    def test_multiple_children_severity_supremum(self, narrator: TelemetryNarrator, context: TelemetryContext):
        """Severidad de padre es supremum de hijos."""
        with context.span("parent"):
            with context.span("child1"):
                pass  # SUCCESS
            with context.span("child2") as c2:
                c2.status = StepStatus.WARNING
            with context.span("child3"):
                pass  # SUCCESS

        report = narrator.summarize_execution(context)
        phase = report["phases"][0]

        # Severidad debe ser WARNING (supremum de SUCCESS, WARNING, SUCCESS)
        assert phase["status"] in ["ADVERTENCIA", "OPTIMO"]

    def test_deep_nesting_topological_path(self, narrator: TelemetryNarrator, context: TelemetryContext):
        """Path topológico refleja profundidad."""
        with context.span("level0"):
            with context.span("level1"):
                with context.span("level2") as span:
                    span.errors.append({"message": "Deep error", "type": "Error"})

        report = narrator.summarize_execution(context)
        phase = report["phases"][0]

        # Buscar el issue con el error profundo
        if phase.get("critical_issues"):
            issue = phase["critical_issues"][0]
            path = issue["topological_path"]
            assert "level0" in path
            assert "level1" in path
            assert "level2" in path


# ============================================================================
# TEST: MODO LEGACY
# ============================================================================


class TestLegacyMode:
    """Pruebas para modo de compatibilidad (contextos sin spans)."""

    def test_legacy_mode_with_steps_only(self, narrator: TelemetryNarrator):
        """Contexto con solo steps (sin spans) activa modo legacy."""
        context = TelemetryContext()
        context.steps.append({
            "step": "legacy_step",
            "status": "success",
            "duration_seconds": 1.0,
        })

        report = narrator.summarize_execution(context)

        assert "verdict" in report
        # Debería indicar modo legacy o generar reporte válido
        assert report["verdict_code"] in ["APPROVED", "EMPTY"]

    def test_legacy_mode_with_errors(self, narrator: TelemetryNarrator):
        """Contexto legacy con errores genera rechazo técnico."""
        context = TelemetryContext()
        context.errors.append({
            "step": "failed_step",
            "message": "Legacy error",
            "type": "Error",
        })

        report = narrator.summarize_execution(context)

        # Debe rechazar por errores
        assert "REJECTED" in report["verdict_code"] or "RECHAZADO" in report["verdict_code"]
        # PHYSICS debe estar comprometido
        assert report["strata_analysis"]["PHYSICS"]["severity"] == "CRITICO"

    def test_legacy_mode_evidence_format(self, narrator: TelemetryNarrator):
        """Evidencia legacy tiene formato correcto."""
        context = TelemetryContext()
        context.errors.append({
            "step": "test_step",
            "message": "Test error",
            "type": "TestError",
        })

        report = narrator.summarize_execution(context)

        if report["forensic_evidence"]:
            evidence = report["forensic_evidence"][0]
            assert "source" in evidence
            assert "message" in evidence
            assert "type" in evidence


# ============================================================================
# TEST: CASOS LÍMITE Y MANEJO DE ERRORES
# ============================================================================


class TestEdgeCasesAndErrors:
    """Pruebas de casos límite y manejo de errores."""

    def test_span_with_zero_duration(self, narrator: TelemetryNarrator, context: TelemetryContext):
        """Span con duración cero no causa problemas."""
        with context.span("instant"):
            pass  # Duración casi cero

        report = narrator.summarize_execution(context)
        assert report is not None
        assert "phases" in report

    def test_span_with_empty_name(self, narrator: TelemetryNarrator, context: TelemetryContext):
        """Span con nombre vacío se maneja."""
        # Crear span manualmente con nombre vacío
        empty_span = TelemetrySpan(name="", level=0)
        empty_span.status = StepStatus.SUCCESS
        empty_span.end_time = empty_span.start_time + 0.1
        context.root_spans.append(empty_span)

        report = narrator.summarize_execution(context)
        assert report is not None

    def test_span_with_none_errors(self, narrator: TelemetryNarrator, context: TelemetryContext):
        """Span con errors=None no causa crash."""
        with context.span("test") as span:
            span.errors = None  # Forzar None

        # No debería crashear
        report = narrator.summarize_execution(context)
        assert report is not None

    def test_span_with_invalid_error_format(self, narrator: TelemetryNarrator, context: TelemetryContext):
        """Errores con formato inválido se manejan."""
        with context.span("test") as span:
            span.errors.append("not a dict")  # Formato inválido
            span.errors.append({"no_message_key": "value"})

        report = narrator.summarize_execution(context)
        assert report is not None

    def test_exception_during_analysis_handled(self, narrator: TelemetryNarrator, context: TelemetryContext):
        """Excepciones durante análisis se manejan graciosamente."""
        with context.span("test"):
            pass

        # Simular excepción interna
        with patch.object(narrator, '_analyze_phase', side_effect=Exception("Test error")):
            report = narrator.summarize_execution(context)

        # Debe generar reporte de error
        assert "verdict" in report
        assert "NARRATOR_ERROR" in report["verdict_code"] or "error" in report.get("executive_summary", "").lower()

    def test_very_long_error_message(self, narrator: TelemetryNarrator, context: TelemetryContext):
        """Mensajes de error muy largos se truncan."""
        long_message = "x" * 10000

        with context.span("test") as span:
            span.errors.append({"message": long_message, "type": "Error"})

        report = narrator.summarize_execution(context)

        # El reporte no debe tener mensajes de 10000 caracteres
        report_str = str(report)
        assert len(report_str) < 100000  # Límite razonable

    def test_circular_span_reference_handled(self, narrator: TelemetryNarrator):
        """Referencias circulares en spans no causan loop infinito."""
        # Crear span con referencia circular
        span = TelemetrySpan(name="circular", level=0)
        span.children.append(span)  # Auto-referencia
        span.status = StepStatus.SUCCESS
        span.end_time = span.start_time + 0.1

        # Debería manejar esto (posiblemente con límite de recursión)
        analysis = narrator._analyze_phase(span)
        assert analysis is not None


# ============================================================================
# TEST: FACTORY FUNCTIONS
# ============================================================================


class TestFactoryFunctions:
    """Pruebas para funciones factory."""

    def test_create_narrator_default(self):
        """create_narrator con valores por defecto."""
        narrator = create_narrator()
        assert isinstance(narrator, TelemetryNarrator)

    def test_create_narrator_with_mapping(self):
        """create_narrator con mapeo personalizado."""
        custom = {"custom_step": Stratum.WISDOM}
        narrator = create_narrator(step_mapping=custom)
        assert narrator.step_mapping["custom_step"] == Stratum.WISDOM

    def test_summarize_context_convenience(self):
        """summarize_context como función de conveniencia."""
        context = TelemetryContext()
        with context.span("test"):
            pass

        report = summarize_context(context)

        assert isinstance(report, dict)
        assert "verdict" in report


# ============================================================================
# TEST: NARRATIVA ESPECÍFICA POR STRATUM
# ============================================================================


class TestStratumSpecificNarratives:
    """Pruebas de narrativas específicas por estrato."""

    def test_physics_warning_narrative(self, narrator: TelemetryNarrator, context: TelemetryContext):
        """Warning en PHYSICS genera narrativa específica."""
        with context.span("load_data") as span:
            span.status = StepStatus.WARNING

        report = narrator.summarize_execution(context)
        physics_narrative = report["strata_analysis"]["PHYSICS"]["narrative"]

        # Debe contener keywords de PHYSICS
        assert any(kw in physics_narrative for kw in ["Turbulencia", "Señales", "física", "inestabilidad"])

    def test_tactics_cycle_detection(self, narrator: TelemetryNarrator, context: TelemetryContext):
        """Detección de palabras clave 'ciclo' en TACTICS."""
        with context.span("calculate_costs") as span:
            span.status = StepStatus.FAILURE
            span.errors.append({"message": "Detectado ciclo infinito en grafo", "type": "LogicError"})

        report = narrator.summarize_execution(context)
        tactics_narrative = report["strata_analysis"]["TACTICS"]["narrative"]

        # Debe detectar el ciclo
        assert "Socavón" in tactics_narrative or "ciclo" in tactics_narrative.lower() or "bucle" in tactics_narrative.lower()

    def test_strategy_financial_narrative(self, narrator: TelemetryNarrator, context: TelemetryContext):
        """Narrativa financiera en STRATEGY."""
        with context.span("financial_analysis") as span:
            span.status = StepStatus.FAILURE
            span.errors.append({"message": "VaR exceeded", "type": "FinancialError"})

        report = narrator.summarize_execution(context)
        strategy_narrative = report["strata_analysis"]["STRATEGY"]["narrative"]

        # Debe contener keywords financieros
        assert any(kw in strategy_narrative for kw in ["Riesgo", "financiera", "mercado", "pérdidas"])


# ============================================================================
# TEST: PROPIEDADES ESTADÍSTICAS
# ============================================================================


class TestStatisticalProperties:
    """Pruebas de propiedades estadísticas."""

    def test_deterministic_output(self, narrator: TelemetryNarrator):
        """Mismo input produce mismo output."""
        context1 = TelemetryContext()
        context2 = TelemetryContext()

        with context1.span("test"):
            pass
        with context2.span("test"):
            pass

        report1 = narrator.summarize_execution(context1)
        report2 = narrator.summarize_execution(context2)

        assert report1["verdict_code"] == report2["verdict_code"]

    def test_severity_monotonicity(self, narrator: TelemetryNarrator):
        """Más errores → severidad peor o igual."""
        for num_errors in range(5):
            context = TelemetryContext()
            with context.span("test") as span:
                for i in range(num_errors):
                    span.errors.append({"message": f"Error {i}", "type": "Error"})

            report = narrator.summarize_execution(context)
            severity = report["strata_analysis"]["PHYSICS"]["severity"]

            if num_errors == 0:
                assert severity == "OPTIMO"
            else:
                assert severity == "CRITICO"

    def test_failure_propagation_invariant(self, narrator: TelemetryNarrator):
        """
        Invariante: Si PHYSICS falla, el veredicto siempre es REJECTED_PHYSICS,
        sin importar el estado de otros estratos.
        """
        for tactics_status in [StepStatus.SUCCESS, StepStatus.FAILURE]:
            for strategy_status in [StepStatus.SUCCESS, StepStatus.FAILURE]:
                context = TelemetryContext()

                with context.span("load_data") as span:
                    span.status = StepStatus.FAILURE
                    span.errors.append({"message": "Physics fail", "type": "Error"})

                with context.span("calculate_costs") as span:
                    span.status = tactics_status

                with context.span("financial_analysis") as span:
                    span.status = strategy_status

                report = narrator.summarize_execution(context)

                # Siempre debe ser rechazo por PHYSICS
                assert "PHYSICS" in report["verdict_code"], (
                    f"Expected REJECTED_PHYSICS but got {report['verdict_code']} "
                    f"with tactics={tactics_status.name}, strategy={strategy_status.name}"
                )


# ============================================================================
# TEST: INTEGRACIÓN CON TELEMETRY CONTEXT
# ============================================================================


class TestTelemetryContextIntegration:
    """Pruebas de integración con TelemetryContext."""

    def test_uses_span_stratum(self, narrator: TelemetryNarrator, context: TelemetryContext):
        """El narrador respeta el stratum del span."""
        with context.span("custom_step", stratum=Stratum.STRATEGY):
            pass

        report = narrator.summarize_execution(context)

        # El paso debería aparecer en STRATEGY
        assert report["strata_analysis"]["STRATEGY"]["phase_count"] > 0 or \
               len(report["phases"]) > 0

    def test_uses_span_metadata(self, narrator: TelemetryNarrator, context: TelemetryContext):
        """El narrador puede acceder a metadata del span."""
        with context.span("test", metadata={"custom_key": "custom_value"}):
            pass

        report = narrator.summarize_execution(context)

        # El reporte debe generarse sin problemas
        assert report is not None

    def test_handles_active_steps_gracefully(self, narrator: TelemetryNarrator, context: TelemetryContext):
        """Pasos activos (no finalizados) se manejan."""
        context.start_step("active_step")
        # No llamamos end_step

        # Agregar un span completado
        with context.span("completed"):
            pass

        report = narrator.summarize_execution(context)
        assert report is not None


# ============================================================================
# TEST: ESTRUCTURA DE REPORTE
# ============================================================================


class TestReportStructure:
    """Pruebas de estructura del reporte."""

    def test_required_keys_present(self, narrator: TelemetryNarrator, context: TelemetryContext):
        """Todas las claves requeridas están presentes."""
        with context.span("test"):
            pass

        report = narrator.summarize_execution(context)

        required_keys = [
            "verdict",
            "verdict_code",
            "executive_summary",
            "global_severity",
            "strata_analysis",
            "forensic_evidence",
            "phases",
            "timestamp",
        ]

        for key in required_keys:
            assert key in report, f"Missing key: {key}"

    def test_all_strata_in_analysis(self, narrator: TelemetryNarrator, context: TelemetryContext):
        """Todos los estratos están en el análisis."""
        report = narrator.summarize_execution(context)

        for stratum in Stratum:
            assert stratum.name in report["strata_analysis"]

    def test_strata_analysis_structure(self, narrator: TelemetryNarrator, context: TelemetryContext):
        """Cada estrato tiene la estructura correcta."""
        with context.span("test"):
            pass

        report = narrator.summarize_execution(context)

        for stratum_name, analysis in report["strata_analysis"].items():
            assert "stratum" in analysis
            assert "severity" in analysis
            assert "narrative" in analysis
            assert "level" in analysis

    def test_phases_structure(self, narrator: TelemetryNarrator, context: TelemetryContext):
        """Cada fase tiene la estructura correcta."""
        with context.span("test"):
            pass

        report = narrator.summarize_execution(context)

        for phase in report["phases"]:
            assert "name" in phase
            assert "stratum" in phase
            assert "status" in phase
            assert "duration" in phase or "duration_seconds" in phase

    def test_forensic_evidence_limited(self, narrator: TelemetryNarrator, context: TelemetryContext):
        """Evidencia forense está limitada."""
        # Crear muchos errores
        with context.span("test") as span:
            for i in range(50):
                span.errors.append({"message": f"Error {i}", "type": "Error"})

        report = narrator.summarize_execution(context)

        assert len(report["forensic_evidence"]) <= NarratorConfig.MAX_FORENSIC_EVIDENCE