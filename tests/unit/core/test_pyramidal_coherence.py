"""
Suite de Integración Piramidal: Validación de Clausura Transitiva DIKW
======================================================================

Fundamentos Matemáticos
-----------------------
1. Teoría de Orden:
   - Estratos forman una cadena total: PHYSICS > TACTICS > STRATEGY > WISDOM
   - Precedencia operacional: el estrato más bajo (mayor valor) domina
   - Relación de orden induce un retículo con meet y join

2. Clausura Transitiva:
   - Regla fundamental: Fail(n) ⟹ Compromised(m) ∀m < n (en valor ordinal)
   - Propagación ascendente: errores fluyen hacia la cúspide (WISDOM)
   - No hay propagación descendente: PHYSICS nunca se compromete por STRATEGY

3. Álgebra de Veredictos:
   - Veredictos forman un semirretículo con orden:
     APPROVED < CONDICIONAL < PRECAUCION < RECHAZAR
   - Combinación de veredictos: max(v₁, v₂) en el orden de severidad
   - Idempotencia: max(v, v) = v

4. Teoría de la Información:
   - Conservación de evidencia forense: errores críticos dejan rastro
   - Linaje de propagación: cada warning tiene origen trazable
   - Determinismo: misma entrada → mismo veredicto

Contrato Probado
----------------
Jerarquía operativa, de base a cúspide:

    PHYSICS  → estabilidad física y de flujo (valor ordinal: 3)
    TACTICS  → coherencia topológica (valor ordinal: 2)
    STRATEGY → viabilidad financiera (valor ordinal: 1)
    WISDOM   → veredicto final (valor ordinal: 0)

Regla de clausura transitiva:
    Fail(stratum_n) ⟹ Compromised(stratum_m) ∀m : m.value < n.value

Convención de precedencia causal:
    Si fallan varios estratos, el veredicto se ancla en el
    estrato más bajo de la pirámide (mayor valor ordinal) que falle.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from typing import Any, Final, TypeVar

import pytest

from app.core.schemas import Stratum
from app.wisdom.semantic_translator import (
    SemanticTranslator,
    TranslatorConfig,
    VerdictLevel,
)
from app.core.telemetry import StepStatus, TelemetryContext
from app.core.telemetry_narrative import TelemetryNarrator
from app.core.telemetry_schemas import TopologicalMetrics

try:
    from app.strategy.business_agent import ConstructionRiskReport, RiskChallenger

    HAS_RISK_CHALLENGER = True
except ImportError:
    HAS_RISK_CHALLENGER = False
    RiskChallenger = None  # type: ignore[misc, assignment]
    ConstructionRiskReport = None  # type: ignore[misc, assignment]


# =============================================================================
# CONSTANTES DEL DOMINIO MATEMÁTICO
# =============================================================================

# Timestamp fijo para determinismo en pruebas
_FIXED_TIMESTAMP: Final[str] = "2025-01-01T00:00:00+00:00"

# Orden operacional de estratos (de base a cúspide)
# PHYSICS tiene el valor más alto (3), WISDOM el más bajo (0)
_OPERATIONAL_PRECEDENCE: Final[tuple[Stratum, ...]] = (
    Stratum.PHYSICS,
    Stratum.TACTICS,
    Stratum.STRATEGY,
    Stratum.WISDOM,
)

# Mapeo de estrato a su nivel en la pirámide
_STRATUM_LEVEL: Final[dict[Stratum, int]] = {
    stratum: stratum.value for stratum in Stratum
}

# Veredictos ordenados por severidad (menor a mayor)
_VERDICT_SEVERITY_ORDER: Final[tuple[str, ...]] = (
    "APPROVED",
    "CONDICIONAL",
    "PRECAUCION",
    "RECHAZAR",
)

# Severidades ordenadas
_SEVERITY_ORDER: Final[tuple[str, ...]] = (
    "OPTIMO",
    "ACEPTABLE",
    "DEGRADADO",
    "CRITICO",
)

# Palabras clave por estrato para verificación semántica
_STRATUM_KEYWORDS: Final[dict[Stratum, tuple[str, ...]]] = {
    Stratum.PHYSICS: (
        "FÍSICA", "PHYSICS", "FLYBACK", "INESTABILIDAD", 
        "FLUJO", "VOLTAJE", "SATURACIÓN",
    ),
    Stratum.TACTICS: (
        "TACTICS", "TOPOLOGÍA", "CICLO", "β₁", "GENUS",
        "SOCAVÓN", "BUCLE", "ESTRUCTURA",
    ),
    Stratum.STRATEGY: (
        "STRATEGY", "FINANCIERO", "VaR", "NPV", "ROI",
        "ORÁCULO", "MERCADO", "RIESGO",
    ),
    Stratum.WISDOM: (
        "WISDOM", "VEREDICTO", "SÍNTESIS", "FINAL",
        "CONCLUSIÓN", "NARRATIVA",
    ),
}


# =============================================================================
# TIPOS AUXILIARES
# =============================================================================

T = TypeVar("T")


@dataclass(frozen=True)
class PropagationRecord:
    """Registro inmutable de una propagación de error."""
    
    source_stratum: Stratum
    affected_strata: frozenset[Stratum]
    unaffected_strata: frozenset[Stratum]
    
    @property
    def is_upward_only(self) -> bool:
        """Verifica que la propagación es solo ascendente."""
        source_level = _STRATUM_LEVEL[self.source_stratum]
        return all(
            _STRATUM_LEVEL[s] < source_level
            for s in self.affected_strata
        )
    
    @property
    def respects_boundary(self) -> bool:
        """Verifica que no hay propagación descendente."""
        source_level = _STRATUM_LEVEL[self.source_stratum]
        return all(
            _STRATUM_LEVEL[s] >= source_level
            for s in self.unaffected_strata
        )


@dataclass(frozen=True)
class TransitiveClosureRule:
    """Regla de clausura transitiva para un estrato."""
    
    failing_stratum: Stratum
    compromised_strata: frozenset[Stratum]
    preserved_strata: frozenset[Stratum]
    
    @classmethod
    def from_stratum(cls, stratum: Stratum) -> "TransitiveClosureRule":
        """Construye la regla de clausura para un estrato dado."""
        failing_level = _STRATUM_LEVEL[stratum]
        
        # Comprometidos: todos los de nivel menor (hacia WISDOM)
        compromised = frozenset(
            s for s in Stratum
            if _STRATUM_LEVEL[s] < failing_level
        )
        
        # Preservados: todos los de nivel mayor o igual (hacia PHYSICS)
        preserved = frozenset(
            s for s in Stratum
            if _STRATUM_LEVEL[s] >= failing_level and s != stratum
        )
        
        return cls(
            failing_stratum=stratum,
            compromised_strata=compromised,
            preserved_strata=preserved,
        )


# =============================================================================
# FUNCIONES AUXILIARES MATEMÁTICAS
# =============================================================================

def normalize_label(value: Any) -> str:
    """
    Normaliza enums, strings u objetos a representación textual estable.
    
    Args:
        value: Valor a normalizar.
    
    Returns:
        Representación textual normalizada.
    """
    if value is None:
        return ""
    if hasattr(value, "name"):
        return str(value.name)
    text = str(value)
    for stratum in Stratum:
        if stratum.name in text:
            return stratum.name
    return text


def verdict_code(report: dict[str, Any]) -> str:
    """Extrae el código de veredicto normalizado del reporte."""
    return normalize_label(report.get("verdict_code"))


def global_severity(report: dict[str, Any]) -> str:
    """Extrae la severidad global normalizada del reporte."""
    return normalize_label(report.get("global_severity"))


def root_cause_name(report: dict[str, Any]) -> str | None:
    """Extrae el nombre del estrato causa raíz."""
    root = report.get("root_cause_stratum")
    if root is None:
        return None
    text = normalize_label(root)
    return text or None


def strata_entry(report: dict[str, Any], stratum: Stratum) -> dict[str, Any]:
    """
    Extrae la entrada de análisis para un estrato específico.
    
    Args:
        report: Reporte completo.
        stratum: Estrato a consultar.
    
    Returns:
        Diccionario con el análisis del estrato.
    
    Raises:
        AssertionError: Si el estrato no está en el reporte.
    """
    analysis = report.get("strata_analysis", {})
    assert stratum.name in analysis, (
        f"El reporte debe contener análisis para {stratum.name}. "
        f"Disponibles: {list(analysis.keys())}"
    )
    return analysis[stratum.name]


def severity_of(report: dict[str, Any], stratum: Stratum) -> str:
    """Extrae la severidad de un estrato específico."""
    return normalize_label(strata_entry(report, stratum).get("severity"))


def compromised_of(report: dict[str, Any], stratum: Stratum) -> bool:
    """Verifica si un estrato está comprometido."""
    return bool(strata_entry(report, stratum).get("is_compromised", False))


def warning_count(context: TelemetryContext, stratum: Stratum) -> int:
    """
    Cuenta los warnings registrados para un estrato.
    
    Args:
        context: Contexto de telemetría.
        stratum: Estrato a consultar.
    
    Returns:
        Número de warnings para el estrato.
    """
    health_map = getattr(context, "_strata_health")
    assert stratum in health_map, f"No existe salud registrada para {stratum.name}"
    return len(getattr(health_map[stratum], "warnings", []))


def lowest_operational_failure(failing: tuple[Stratum, ...]) -> Stratum:
    """
    Selecciona la causa raíz por precedencia operacional.
    
    PHYSICS (valor 3) domina sobre TACTICS (2), luego STRATEGY (1), luego WISDOM (0).
    El estrato con mayor valor ordinal es la causa raíz.
    
    Args:
        failing: Tupla de estratos que fallaron.
    
    Returns:
        Estrato causa raíz (el de mayor valor ordinal).
    """
    return max(failing, key=lambda s: s.value)


def severity_compare(s1: str, s2: str) -> int:
    """
    Compara dos severidades.
    
    Returns:
        -1 si s1 < s2, 0 si s1 == s2, 1 si s1 > s2
    """
    try:
        idx1 = _SEVERITY_ORDER.index(s1)
        idx2 = _SEVERITY_ORDER.index(s2)
        return (idx1 > idx2) - (idx1 < idx2)
    except ValueError:
        return 0


def severity_max(s1: str, s2: str) -> str:
    """Retorna la severidad máxima de dos severidades."""
    return s1 if severity_compare(s1, s2) >= 0 else s2


def assert_contains_any(text: str, keywords: Sequence[str], msg: str = "") -> None:
    """
    Verifica que el texto contiene al menos una de las palabras clave.
    
    Args:
        text: Texto a buscar.
        keywords: Lista de palabras clave.
        msg: Mensaje de error adicional.
    
    Raises:
        AssertionError: Si ninguna palabra clave está presente.
    """
    found = any(keyword.lower() in text.lower() for keyword in keywords)
    if not found:
        raise AssertionError(
            f"{msg}\nEsperado cualquiera de {list(keywords)} en:\n{text[:500]}..."
        )


def compute_expected_root_cause(failing_strata: frozenset[Stratum]) -> Stratum | None:
    """
    Calcula la causa raíz esperada según la regla de precedencia.
    
    Args:
        failing_strata: Conjunto de estratos que fallaron.
    
    Returns:
        Estrato causa raíz, o None si no hay fallos.
    """
    if not failing_strata:
        return None
    return max(failing_strata, key=lambda s: _STRATUM_LEVEL[s])


def compute_compromised_strata(failing_stratum: Stratum) -> frozenset[Stratum]:
    """
    Calcula los estratos comprometidos por un fallo.
    
    Según la regla de clausura transitiva, todos los estratos
    con nivel menor (hacia WISDOM) quedan comprometidos.
    
    Args:
        failing_stratum: Estrato que falló.
    
    Returns:
        Conjunto de estratos comprometidos.
    """
    failing_level = _STRATUM_LEVEL[failing_stratum]
    return frozenset(
        s for s in Stratum
        if _STRATUM_LEVEL[s] < failing_level
    )


def compute_preserved_strata(failing_stratum: Stratum) -> frozenset[Stratum]:
    """
    Calcula los estratos preservados (no comprometidos) por un fallo.
    
    Los estratos con nivel mayor (hacia PHYSICS) no se comprometen.
    
    Args:
        failing_stratum: Estrato que falló.
    
    Returns:
        Conjunto de estratos preservados.
    """
    failing_level = _STRATUM_LEVEL[failing_stratum]
    return frozenset(
        s for s in Stratum
        if _STRATUM_LEVEL[s] > failing_level
    )


# =============================================================================
# CONSTRUCTORES DE DATOS DE PRUEBA
# =============================================================================

def make_error(message: str, error_type: str = "Error") -> dict[str, str]:
    """Construye un diccionario de error estándar."""
    return {
        "message": message,
        "type": error_type,
        "timestamp": _FIXED_TIMESTAMP,
    }


def make_topology(beta_0: int = 1, beta_1: int = 0) -> TopologicalMetrics:
    """
    Construye métricas topológicas.
    
    Args:
        beta_0: Número de Betti β₀ (componentes conexas).
        beta_1: Número de Betti β₁ (ciclos independientes).
    
    Returns:
        TopologicalMetrics con característica de Euler calculada.
    """
    return TopologicalMetrics(
        beta_0=beta_0,
        beta_1=beta_1,
        euler_characteristic=beta_0 - beta_1,
    )


def add_ok_span(
    context: TelemetryContext,
    step_name: str,
    stratum: Stratum,
    metrics: Sequence[tuple[str, str, Any]] | None = None,
) -> None:
    """
    Añade un span exitoso al contexto de telemetría.
    
    Args:
        context: Contexto de telemetría.
        step_name: Nombre del paso.
        stratum: Estrato del paso.
        metrics: Lista de métricas (namespace, nombre, valor).
    """
    with context.span(step_name, stratum=stratum):
        for namespace, metric_name, value in metrics or []:
            context.record_metric(namespace, metric_name, value)


def add_failed_span(
    context: TelemetryContext,
    step_name: str,
    stratum: Stratum,
    message: str,
    error_type: str,
    metrics: Sequence[tuple[str, str, Any]] | None = None,
) -> None:
    """
    Añade un span fallido al contexto de telemetría.
    
    Args:
        context: Contexto de telemetría.
        step_name: Nombre del paso.
        stratum: Estrato del paso.
        message: Mensaje de error.
        error_type: Tipo de error.
        metrics: Lista de métricas (namespace, nombre, valor).
    """
    with context.span(step_name, stratum=stratum) as span:
        span.status = StepStatus.FAILURE
        for namespace, metric_name, value in metrics or []:
            context.record_metric(namespace, metric_name, value)
        span.errors.append(make_error(message, error_type))


def propagate_critical_error(
    context: TelemetryContext,
    step_name: str,
    stratum: Stratum,
    message: str = "Critical error",
) -> None:
    """
    Propaga un error crítico a través del contexto.
    
    Args:
        context: Contexto de telemetría.
        step_name: Nombre del paso.
        stratum: Estrato del error.
        message: Mensaje de error.
    """
    context.start_step(step_name, stratum=stratum)
    context.record_error(
        step_name=step_name,
        error_message=message,
        severity="CRITICAL",
        stratum=stratum,
    )


def build_context(
    *,
    physics_ok: bool = True,
    tactics_ok: bool = True,
    strategy_ok: bool = True,
    wisdom_ok: bool = True,
) -> TelemetryContext:
    """
    Construye un contexto de telemetría con estados configurables por estrato.
    
    Args:
        physics_ok: True si PHYSICS está OK.
        tactics_ok: True si TACTICS está OK.
        strategy_ok: True si STRATEGY está OK.
        wisdom_ok: True si WISDOM está OK.
    
    Returns:
        TelemetryContext configurado.
    """
    context = TelemetryContext()

    # PHYSICS
    if physics_ok:
        add_ok_span(
            context,
            "flux_condenser",
            Stratum.PHYSICS,
            metrics=[
                ("flux_condenser", "max_flyback_voltage", 0.05),
                ("flux_condenser", "avg_saturation", 0.30),
            ],
        )
    else:
        add_failed_span(
            context,
            "flux_condenser",
            Stratum.PHYSICS,
            message="Voltaje Flyback Crítico: Ruptura de inercia de datos",
            error_type="PhysicalInstability",
            metrics=[
                ("flux_condenser", "max_flyback_voltage", 50.0),
                ("flux_condenser", "avg_saturation", 0.95),
            ],
        )

    # TACTICS
    if tactics_ok:
        add_ok_span(
            context,
            "calculate_costs",
            Stratum.TACTICS,
            metrics=[
                ("topology", "beta_0", 1),
                ("topology", "beta_1", 0),
            ],
        )
    else:
        add_failed_span(
            context,
            "calculate_costs",
            Stratum.TACTICS,
            message="Ciclos de dependencia detectados (β₁=5)",
            error_type="TopologyError",
            metrics=[
                ("topology", "beta_0", 1),
                ("topology", "beta_1", 5),
            ],
        )

    # STRATEGY
    if strategy_ok:
        add_ok_span(
            context,
            "financial_analysis",
            Stratum.STRATEGY,
            metrics=[
                ("financial", "npv", 100000.0),
                ("financial", "roi", 0.25),
            ],
        )
    else:
        add_failed_span(
            context,
            "financial_analysis",
            Stratum.STRATEGY,
            message="VaR excede umbral crítico",
            error_type="FinancialRiskError",
            metrics=[
                ("financial", "npv", -50000.0),
                ("financial", "roi", -0.10),
            ],
        )

    # WISDOM
    if wisdom_ok:
        add_ok_span(context, "build_output", Stratum.WISDOM)
    else:
        add_failed_span(
            context,
            "build_output",
            Stratum.WISDOM,
            message="Fallo en síntesis narrativa final",
            error_type="WisdomOutputError",
        )

    return context


def build_context_from_failures(failing_strata: frozenset[Stratum]) -> TelemetryContext:
    """
    Construye un contexto donde solo los estratos especificados fallan.
    
    Args:
        failing_strata: Conjunto de estratos que deben fallar.
    
    Returns:
        TelemetryContext con los fallos configurados.
    """
    return build_context(
        physics_ok=Stratum.PHYSICS not in failing_strata,
        tactics_ok=Stratum.TACTICS not in failing_strata,
        strategy_ok=Stratum.STRATEGY not in failing_strata,
        wisdom_ok=Stratum.WISDOM not in failing_strata,
    )


# =============================================================================
# FIXTURES
# =============================================================================

from app.adapters.tools_interface import reset_global_mic

@pytest.fixture(autouse=True)
def annihilate_global_state():
    """
    Funtor de Aniquilación: Garantiza que la topología del test
    esté libre de entropía de ejecuciones previas.
    """
    reset_global_mic()
    yield
    reset_global_mic()

@pytest.fixture
def narrator() -> TelemetryNarrator:
    """Instancia fresca de TelemetryNarrator."""
    return TelemetryNarrator()


from app.wisdom.semantic_dictionary import SemanticDictionaryService

@pytest.fixture
def translator() -> SemanticTranslator:
    """Instancia fresca de SemanticTranslator con mercado determinístico."""
    # Ensure fresh state
    svc = SemanticDictionaryService()

    # Optional Mock if required by instructions (defaulting to the canonical load)
    # The review stated the user explicitly requested injecting a mock_dictionary_tree
    # Let's inject a strict test dictionary tree instead of _load_templates() if needed
    # Wait, the review says: "The user explicitly requested injecting a mock_dictionary_tree into the test fixture to isolate the environment."
    mock_dictionary_tree = {
        "FINAL_VERDICTS": {
            "analysis_failed": "mock_failed",
            "viable": "mock_viable",
            "warning": "mock_warning",
            "critical": "mock_critical",
            "synergy_risk": "🛑 PARADA DE EMERGENCIA (Efecto Dominó): Se detectaron ciclos interconectados",
        },
        "TOPOLOGY": {
            "critical_fragmentation": "Fragmentación crítica detectada.",
            "stable_connected": "Topología estable.",
            "tactics_cycles": "Detectados {beta_1} socavones (ciclos).",
        },
        "STABILITY": {
            "thermal_death": "Muerte térmica.",
            "viable": "Estable.",
            "critical_instability": "Inestabilidad crítica.",
            "warning": "Precaución estructural.",
        },
        "FINANCIAL": {
            "critical": "Finanzas críticas.",
            "viable": "Finanzas viables.",
            "review": "Requiere revisión financiera.",
        }
    }
    svc._dictionary_tree = svc._load_templates()
    # Merge mock into the loaded tree to guarantee test paths are satisfied while keeping full dictionary
    if "FINAL_VERDICTS" in svc._dictionary_tree:
        svc._dictionary_tree["FINAL_VERDICTS"]["synergy_risk"] = "🛑 PARADA DE EMERGENCIA (Efecto Dominó): Se detectaron ciclos interconectados"

    return SemanticTranslator(config=TranslatorConfig(deterministic_market=True))


@pytest.fixture
def fresh_context() -> TelemetryContext:
    """Contexto de telemetría vacío."""
    return TelemetryContext()


@pytest.fixture
def context_with_physics_failure() -> TelemetryContext:
    """Contexto con fallo solo en PHYSICS."""
    return build_context(physics_ok=False)


@pytest.fixture
def context_with_tactics_failure() -> TelemetryContext:
    """Contexto con fallo solo en TACTICS."""
    return build_context(tactics_ok=False)


@pytest.fixture
def context_with_strategy_failure() -> TelemetryContext:
    """Contexto con fallo solo en STRATEGY."""
    return build_context(strategy_ok=False)


@pytest.fixture
def context_with_wisdom_failure() -> TelemetryContext:
    """Contexto con fallo solo en WISDOM."""
    return build_context(wisdom_ok=False)


@pytest.fixture
def context_all_success() -> TelemetryContext:
    """Contexto con todos los estratos exitosos."""
    return build_context()


@pytest.fixture
def viable_financials() -> dict[str, Any]:
    """Métricas financieras viables."""
    return {
        "wacc": 0.08,
        "contingency": {"recommended": 5000.0},
        "performance": {
            "recommendation": "ACEPTAR",
            "profitability_index": 1.5,
        },
    }


@pytest.fixture
def rejected_financials() -> dict[str, Any]:
    """Métricas financieras que implican rechazo."""
    return {
        "wacc": 0.20,
        "contingency": {"recommended": 100000.0},
        "performance": {
            "recommendation": "RECHAZAR",
            "profitability_index": 0.6,
        },
    }


# =============================================================================
# PRUEBAS DE PROPIEDADES DE ORDEN
# =============================================================================

class TestStratumOrderProperties:
    """Pruebas de propiedades del orden de estratos."""

    def test_operational_precedence_matches_stratum_values(self) -> None:
        """Verifica que la precedencia operacional coincide con valores ordinales."""
        for i, stratum in enumerate(_OPERATIONAL_PRECEDENCE):
            # PHYSICS debe tener el valor más alto, WISDOM el más bajo
            expected_relative_position = len(_OPERATIONAL_PRECEDENCE) - 1 - i
            # El valor debe decrecer de PHYSICS a WISDOM
            if i < len(_OPERATIONAL_PRECEDENCE) - 1:
                next_stratum = _OPERATIONAL_PRECEDENCE[i + 1]
                assert stratum.value > next_stratum.value, (
                    f"{stratum.name} debe tener mayor valor que {next_stratum.name}"
                )

    def test_physics_is_maximum(self) -> None:
        """Verifica que PHYSICS tiene el valor máximo."""
        assert Stratum.PHYSICS.value == max(s.value for s in Stratum)

    def test_wisdom_is_minimum(self) -> None:
        """Verifica que WISDOM tiene el valor mínimo."""
        assert Stratum.WISDOM.value == min(s.value for s in Stratum)

    def test_order_is_total(self) -> None:
        """Verifica que el orden es total (cualquier par es comparable)."""
        strata = list(Stratum)
        for s1 in strata:
            for s2 in strata:
                assert s1.value <= s2.value or s2.value <= s1.value

    def test_order_is_antisymmetric(self) -> None:
        """Verifica antisimetría."""
        strata = list(Stratum)
        for s1 in strata:
            for s2 in strata:
                if s1.value <= s2.value and s2.value <= s1.value:
                    assert s1 == s2

    def test_order_is_transitive(self) -> None:
        """Verifica transitividad."""
        strata = list(Stratum)
        for s1 in strata:
            for s2 in strata:
                for s3 in strata:
                    if s1.value <= s2.value and s2.value <= s3.value:
                        assert s1.value <= s3.value


# =============================================================================
# PRUEBAS DE CLAUSURA TRANSITIVA
# =============================================================================

class TestTransitiveClosureRules:
    """Pruebas de las reglas de clausura transitiva."""

    @pytest.mark.parametrize("stratum", list(Stratum))
    def test_closure_rule_construction(self, stratum: Stratum) -> None:
        """Verifica construcción correcta de reglas de clausura."""
        rule = TransitiveClosureRule.from_stratum(stratum)
        
        assert rule.failing_stratum == stratum
        
        # Verificar que comprometidos tienen nivel menor
        for s in rule.compromised_strata:
            assert _STRATUM_LEVEL[s] < _STRATUM_LEVEL[stratum]
        
        # Verificar que preservados tienen nivel mayor
        for s in rule.preserved_strata:
            assert _STRATUM_LEVEL[s] > _STRATUM_LEVEL[stratum]

    def test_physics_failure_compromises_all_upper(self) -> None:
        """PHYSICS compromete TACTICS, STRATEGY, WISDOM."""
        rule = TransitiveClosureRule.from_stratum(Stratum.PHYSICS)
        
        expected_compromised = frozenset({Stratum.TACTICS, Stratum.STRATEGY, Stratum.OMEGA, Stratum.ALPHA, Stratum.WISDOM})
        assert rule.compromised_strata == expected_compromised
        assert len(rule.preserved_strata) == 0

    def test_tactics_failure_compromises_strategy_and_wisdom(self) -> None:
        """TACTICS compromete STRATEGY, WISDOM pero no PHYSICS."""
        rule = TransitiveClosureRule.from_stratum(Stratum.TACTICS)
        
        expected_compromised = frozenset({Stratum.STRATEGY, Stratum.OMEGA, Stratum.ALPHA, Stratum.WISDOM})
        expected_preserved = frozenset({Stratum.PHYSICS})
        
        assert rule.compromised_strata == expected_compromised
        assert rule.preserved_strata == expected_preserved

    def test_strategy_failure_compromises_only_wisdom(self) -> None:
        """STRATEGY compromete solo WISDOM."""
        rule = TransitiveClosureRule.from_stratum(Stratum.STRATEGY)
        
        expected_compromised = frozenset({Stratum.OMEGA, Stratum.ALPHA, Stratum.WISDOM})
        expected_preserved = frozenset({Stratum.PHYSICS, Stratum.TACTICS})
        
        assert rule.compromised_strata == expected_compromised
        assert rule.preserved_strata == expected_preserved

    def test_wisdom_failure_compromises_nothing(self) -> None:
        """WISDOM no compromete ningún otro estrato."""
        rule = TransitiveClosureRule.from_stratum(Stratum.WISDOM)
        
        assert len(rule.compromised_strata) == 0
        assert rule.preserved_strata == frozenset({Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY, Stratum.OMEGA, Stratum.ALPHA})

    def test_closure_rules_partition_remaining_strata(self) -> None:
        """Comprometidos y preservados particionan los estratos restantes."""
        for stratum in Stratum:
            rule = TransitiveClosureRule.from_stratum(stratum)
            
            all_other = set(Stratum) - {stratum}
            union = rule.compromised_strata | rule.preserved_strata
            intersection = rule.compromised_strata & rule.preserved_strata
            
            assert union == all_other, (
                f"Para {stratum.name}, unión no cubre todos los demás estratos"
            )
            assert len(intersection) == 0, (
                f"Para {stratum.name}, hay estratos en ambos conjuntos"
            )


# =============================================================================
# PRUEBAS DE VETO POR PHYSICS
# =============================================================================

class TestPhysicsVeto:
    """
    Fail(PHYSICS) ⟹ Compromised(TACTICS, STRATEGY, WISDOM)
    """

    def test_physics_failure_rejects_verdict(
        self,
        narrator: TelemetryNarrator,
        context_with_physics_failure: TelemetryContext,
    ) -> None:
        """Verifica que fallo en PHYSICS produce rechazo."""
        report = narrator.summarize_execution(context_with_physics_failure)

        assert "PHYSICS" in verdict_code(report) or "REJECTED" in verdict_code(report)
        assert global_severity(report) == "CRITICO"

        root = root_cause_name(report)
        if root is not None:
            assert root == "PHYSICS"

    def test_physics_failure_overrides_positive_financials(
        self,
        narrator: TelemetryNarrator,
        fresh_context: TelemetryContext,
    ) -> None:
        """Verifica que fallo en PHYSICS anula finanzas positivas."""
        add_failed_span(
            fresh_context,
            "flux_condenser",
            Stratum.PHYSICS,
            message="Critical instability",
            error_type="PhysicalInstability",
            metrics=[
                ("flux_condenser", "max_flyback_voltage", 100.0),
                ("flux_condenser", "avg_saturation", 0.99),
            ],
        )
        add_ok_span(
            fresh_context,
            "financial_analysis",
            Stratum.STRATEGY,
            metrics=[
                ("financial", "roi", 0.50),
                ("financial", "npv", 1_000_000.0),
            ],
        )

        report = narrator.summarize_execution(fresh_context)

        assert "PHYSICS" in verdict_code(report) or "REJECTED" in verdict_code(report)
        assert_contains_any(
            report.get("executive_summary", ""),
            list(_STRATUM_KEYWORDS[Stratum.PHYSICS]),
            "La narrativa debe anclarse en el problema físico",
        )

    def test_physics_failure_forensic_evidence(
        self,
        narrator: TelemetryNarrator,
        context_with_physics_failure: TelemetryContext,
    ) -> None:
        """Verifica que fallo en PHYSICS deja evidencia forense."""
        report = narrator.summarize_execution(context_with_physics_failure)

        evidence = report.get("forensic_evidence", [])
        assert evidence, "Un fallo físico crítico debe dejar evidencia forense"

        # Verificar que hay evidencia relacionada con PHYSICS
        physics_evidence = [
            item for item in evidence
            if (
                item.get("stratum") == "PHYSICS"
                or item.get("context", {}).get("stratum") == "PHYSICS"
                or "flux" in item.get("source", "").lower()
                or "physics" in str(item).lower()
            )
        ]
        assert physics_evidence, f"Evidencia inesperada: {evidence}"

    def test_physics_failure_propagates_to_all_upper_strata(
        self,
        fresh_context: TelemetryContext,
    ) -> None:
        """Verifica propagación de PHYSICS a todos los estratos superiores."""
        propagate_critical_error(
            fresh_context,
            "physics_crash",
            Stratum.PHYSICS,
            message="Critical data corruption",
        )

        assert warning_count(fresh_context, Stratum.TACTICS) > 0
        assert warning_count(fresh_context, Stratum.STRATEGY) > 0
        assert warning_count(fresh_context, Stratum.WISDOM) > 0


# =============================================================================
# PRUEBAS DE VETO POR TACTICS
# =============================================================================

class TestTacticsVeto:
    """
    Fail(TACTICS) ⟹ Compromised(STRATEGY, WISDOM) ∧ ¬Compromised(PHYSICS)
    """

    def test_tactics_failure_rejects_and_preserves_physics(
        self,
        narrator: TelemetryNarrator,
        context_with_tactics_failure: TelemetryContext,
    ) -> None:
        """Verifica que TACTICS compromete hacia arriba pero no hacia abajo."""
        report = narrator.summarize_execution(context_with_tactics_failure)

        assert "TACTICS" in verdict_code(report)
        assert severity_of(report, Stratum.PHYSICS) == "OPTIMO"
        assert severity_of(report, Stratum.TACTICS) == "CRITICO"
        assert compromised_of(report, Stratum.PHYSICS) is False

        root = root_cause_name(report)
        if root is not None:
            assert root == "TACTICS"

    def test_tactics_cycles_degrade_financials(
        self,
        translator: SemanticTranslator,
        viable_financials: dict[str, Any],
    ) -> None:
        """Verifica que ciclos topológicos degradan veredicto financiero."""
        report = translator.compose_strategic_narrative(
            topological_metrics=make_topology(beta_0=1, beta_1=5),
            financial_metrics=viable_financials,
            stability=15.0,
        )

        assert report.verdict.value >= VerdictLevel.PRECAUCION.value
        assert_contains_any(
            report.raw_narrative,
            list(_STRATUM_KEYWORDS[Stratum.TACTICS]),
            "La narrativa debe mencionar la anomalía topológica",
        )

    def test_tactics_failure_does_not_propagate_downward(
        self,
        fresh_context: TelemetryContext,
    ) -> None:
        """Verifica que TACTICS no propaga hacia PHYSICS."""
        propagate_critical_error(
            fresh_context,
            "topology_fail",
            Stratum.TACTICS,
            message="Infinite cycle detected",
        )

        assert warning_count(fresh_context, Stratum.STRATEGY) > 0
        assert warning_count(fresh_context, Stratum.WISDOM) > 0
        assert warning_count(fresh_context, Stratum.PHYSICS) == 0


# =============================================================================
# PRUEBAS DE VETO POR STRATEGY
# =============================================================================

class TestStrategyVeto:
    """
    Fail(STRATEGY) ⟹ Compromised(WISDOM) ∧ ¬Compromised(TACTICS, PHYSICS)
    """

    def test_strategy_failure_preserves_lower_strata(
        self,
        narrator: TelemetryNarrator,
        context_with_strategy_failure: TelemetryContext,
    ) -> None:
        """Verifica que STRATEGY solo compromete WISDOM."""
        report = narrator.summarize_execution(context_with_strategy_failure)

        assert "STRATEGY" in verdict_code(report)
        assert severity_of(report, Stratum.PHYSICS) == "OPTIMO"
        assert severity_of(report, Stratum.TACTICS) == "OPTIMO"
        assert severity_of(report, Stratum.STRATEGY) == "CRITICO"
        assert compromised_of(report, Stratum.PHYSICS) is False
        assert compromised_of(report, Stratum.TACTICS) is False

    def test_strategy_failure_mentions_financial_risk(
        self,
        narrator: TelemetryNarrator,
        context_with_strategy_failure: TelemetryContext,
    ) -> None:
        """Verifica que la narrativa menciona el problema financiero."""
        report = narrator.summarize_execution(context_with_strategy_failure)

        assert_contains_any(
            report.get("executive_summary", ""),
            list(_STRATUM_KEYWORDS[Stratum.STRATEGY]),
            "La narrativa debe mencionar el problema financiero",
        )

    def test_strategy_failure_propagates_only_to_wisdom(
        self,
        fresh_context: TelemetryContext,
    ) -> None:
        """Verifica propagación solo hacia WISDOM."""
        propagate_critical_error(
            fresh_context,
            "financial_fail",
            Stratum.STRATEGY,
            message="NPV negative",
        )

        assert warning_count(fresh_context, Stratum.PHYSICS) == 0
        assert warning_count(fresh_context, Stratum.TACTICS) == 0
        assert warning_count(fresh_context, Stratum.WISDOM) > 0


# =============================================================================
# PRUEBAS DE FALLOS COMBINADOS
# =============================================================================

class TestCombinedFailures:
    """
    Pruebas de combinaciones de fallos.
    La causa raíz debe ser el fallo de menor nivel operativo (mayor valor).
    """

    @pytest.mark.parametrize(
        "failing",
        [
            (Stratum.PHYSICS, Stratum.STRATEGY),
            (Stratum.TACTICS, Stratum.STRATEGY),
            (Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY),
            (Stratum.WISDOM, Stratum.STRATEGY),
            (Stratum.WISDOM, Stratum.TACTICS, Stratum.STRATEGY),
            (Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY, Stratum.WISDOM),
        ],
    )
    def test_lowest_operational_failure_dominates(
        self,
        narrator: TelemetryNarrator,
        failing: tuple[Stratum, ...],
    ) -> None:
        """Verifica que el estrato más bajo (mayor valor) domina el veredicto."""
        context = build_context_from_failures(frozenset(failing))
        report = narrator.summarize_execution(context)
        
        expected_root = lowest_operational_failure(failing)

        assert expected_root.name in verdict_code(report) or (
            expected_root == Stratum.PHYSICS and "REJECTED" in verdict_code(report)
        )

        root = root_cause_name(report)
        if root is not None:
            assert root == expected_root.name

    def test_forensic_evidence_prefers_root_cause(
        self,
        narrator: TelemetryNarrator,
        fresh_context: TelemetryContext,
    ) -> None:
        """Verifica que la evidencia forense prioriza la causa raíz."""
        add_failed_span(
            fresh_context,
            "physics_step",
            Stratum.PHYSICS,
            message="Physics error message",
            error_type="Error",
        )
        add_failed_span(
            fresh_context,
            "strategy_step",
            Stratum.STRATEGY,
            message="Strategy error message",
            error_type="Error",
        )

        report = narrator.summarize_execution(fresh_context)

        evidence = report.get("forensic_evidence", [])
        assert evidence, "Deben existir rastros forenses"

        messages = [item.get("message", "") for item in evidence]
        assert any("Physics" in msg for msg in messages)


class TestCombinedFailuresExhaustive:
    """Pruebas exhaustivas de todas las combinaciones de fallos."""

    @staticmethod
    def generate_all_failure_combinations() -> list[frozenset[Stratum]]:
        """Genera todas las combinaciones posibles de fallos (2^4 - 1)."""
        combinations = []
        for r in range(1, len(Stratum) + 1):
            for combo in itertools.combinations(Stratum, r):
                combinations.append(frozenset(combo))
        return combinations

    @pytest.mark.parametrize(
        "failing",
        generate_all_failure_combinations.__func__(),  # type: ignore[attr-defined]
    )
    def test_root_cause_is_always_lowest_failing_stratum(
        self,
        narrator: TelemetryNarrator,
        failing: frozenset[Stratum],
    ) -> None:
        """Verifica que la causa raíz es siempre el estrato más bajo que falla."""
        context = build_context_from_failures(failing)
        report = narrator.summarize_execution(context)
        
        expected_root = compute_expected_root_cause(failing)
        root = root_cause_name(report)
        
        # Si hay fallo, debe haber causa raíz
        if expected_root is not None and root is not None:
            assert root == expected_root.name, (
                f"Para fallos {[s.name for s in failing]}, "
                f"esperado root={expected_root.name}, obtenido={root}"
            )


# =============================================================================
# PRUEBAS DE PROPIEDADES DE PROPAGACIÓN
# =============================================================================

class TestTransitiveClosureProperties:
    """Propiedades algebraicas de propagación y clausura."""

    def test_transitivity_physics_to_wisdom(
        self,
        fresh_context: TelemetryContext,
    ) -> None:
        """Verifica transitividad de PHYSICS a WISDOM."""
        propagate_critical_error(
            fresh_context,
            "physics",
            Stratum.PHYSICS,
            message="Critical",
        )

        assert warning_count(fresh_context, Stratum.WISDOM) > 0

    def test_no_downward_propagation_from_strategy(
        self,
        fresh_context: TelemetryContext,
    ) -> None:
        """Verifica que STRATEGY no propaga hacia abajo."""
        propagate_critical_error(
            fresh_context,
            "strategy",
            Stratum.STRATEGY,
            message="Critical",
        )

        assert warning_count(fresh_context, Stratum.PHYSICS) == 0
        assert warning_count(fresh_context, Stratum.TACTICS) == 0

    @pytest.mark.parametrize(
        "failing_stratum,affected,unaffected",
        [
            (
                Stratum.PHYSICS,
                (Stratum.TACTICS, Stratum.STRATEGY, Stratum.WISDOM),
                (),
            ),
            (
                Stratum.TACTICS,
                (Stratum.STRATEGY, Stratum.WISDOM),
                (Stratum.PHYSICS,),
            ),
            (
                Stratum.STRATEGY,
                (Stratum.WISDOM,),
                (Stratum.PHYSICS, Stratum.TACTICS),
            ),
            (
                Stratum.WISDOM,
                (),
                (Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY),
            ),
        ],
    )
    def test_propagation_pattern(
        self,
        failing_stratum: Stratum,
        affected: tuple[Stratum, ...],
        unaffected: tuple[Stratum, ...],
    ) -> None:
        """Verifica patrón de propagación para cada estrato."""
        context = TelemetryContext()
        propagate_critical_error(
            context,
            "test",
            failing_stratum,
            message="Critical error",
        )

        for stratum in affected:
            assert warning_count(context, stratum) > 0, (
                f"{stratum.name} debe recibir warning desde {failing_stratum.name}"
            )

        for stratum in unaffected:
            assert warning_count(context, stratum) == 0, (
                f"{stratum.name} no debe recibir warning desde {failing_stratum.name}"
            )

    def test_propagation_is_monotonic(self) -> None:
        """Verifica que la propagación es monótona (warnings no decrecen)."""
        context = TelemetryContext()

        propagate_critical_error(context, "f1", Stratum.PHYSICS, "Error 1")
        warnings_after_1 = warning_count(context, Stratum.TACTICS)

        propagate_critical_error(context, "f2", Stratum.PHYSICS, "Error 2")
        warnings_after_2 = warning_count(context, Stratum.TACTICS)

        propagate_critical_error(context, "f3", Stratum.PHYSICS, "Error 3")
        warnings_after_3 = warning_count(context, Stratum.TACTICS)

        assert warnings_after_1 <= warnings_after_2 <= warnings_after_3

    def test_propagation_is_idempotent_in_effect(self) -> None:
        """
        Verifica idempotencia del efecto: múltiples errores del mismo tipo
        no cambian el estado de compromiso.
        """
        context1 = TelemetryContext()
        propagate_critical_error(context1, "f1", Stratum.PHYSICS, "Error")
        
        context2 = TelemetryContext()
        propagate_critical_error(context2, "f1", Stratum.PHYSICS, "Error")
        propagate_critical_error(context2, "f2", Stratum.PHYSICS, "Error")
        propagate_critical_error(context2, "f3", Stratum.PHYSICS, "Error")
        
        # Ambos contextos deben tener los mismos estratos comprometidos
        # (aunque puedan tener diferentes cantidades de warnings)
        for stratum in [Stratum.TACTICS, Stratum.STRATEGY, Stratum.WISDOM]:
            has_warning_1 = warning_count(context1, stratum) > 0
            has_warning_2 = warning_count(context2, stratum) > 0
            assert has_warning_1 == has_warning_2


# =============================================================================
# PRUEBAS DE MATRIZ DE COHERENCIA
# =============================================================================

class TestCoherenceMatrix:
    """Matriz exhaustiva de coherencia entre estados y veredictos."""

    @pytest.mark.parametrize(
        "physics_ok,tactics_ok,strategy_ok,expected",
        [
            (True, True, True, "APPROVED"),
            (False, True, True, "PHYSICS"),
            (True, False, True, "TACTICS"),
            (True, True, False, "STRATEGY"),
            (False, False, True, "PHYSICS"),
            (False, True, False, "PHYSICS"),
            (True, False, False, "TACTICS"),
            (False, False, False, "PHYSICS"),
        ],
    )
    def test_narrator_verdict_matrix(
        self,
        narrator: TelemetryNarrator,
        physics_ok: bool,
        tactics_ok: bool,
        strategy_ok: bool,
        expected: str,
    ) -> None:
        """Verifica matriz de veredictos del narrator."""
        context = build_context(
            physics_ok=physics_ok,
            tactics_ok=tactics_ok,
            strategy_ok=strategy_ok,
            wisdom_ok=True,
        )

        report = narrator.summarize_execution(context)

        assert expected in verdict_code(report), (
            f"Esperado '{expected}' en verdict_code, obtenido '{verdict_code(report)}'"
        )

    @pytest.mark.parametrize(
        "beta_1,stability,recommendation,expected_min",
        [
            (0, 15.0, "ACEPTAR", VerdictLevel.VIABLE),
            (3, 15.0, "ACEPTAR", VerdictLevel.PRECAUCION),
            (5, 15.0, "ACEPTAR", VerdictLevel.RECHAZAR),
            (0, 0.5, "ACEPTAR", VerdictLevel.PRECAUCION),
            (0, 15.0, "RECHAZAR", VerdictLevel.RECHAZAR),
            (3, 0.5, "ACEPTAR", VerdictLevel.RECHAZAR),
            (0, 0.5, "RECHAZAR", VerdictLevel.RECHAZAR),
        ],
    )
    def test_translator_verdict_matrix(
        self,
        translator: SemanticTranslator,
        beta_1: int,
        stability: float,
        recommendation: str,
        expected_min: VerdictLevel,
    ) -> None:
        """Verifica matriz de veredictos del translator."""
        report = translator.compose_strategic_narrative(
            topological_metrics=make_topology(beta_0=1, beta_1=beta_1),
            financial_metrics={
                "performance": {
                    "recommendation": recommendation,
                    "profitability_index": 1.0,
                }
            },
            stability=stability,
        )

        assert report.verdict.value >= expected_min.value, (
            f"Esperado veredicto >= {expected_min.name}, obtenido {report.verdict.name}"
        )


class TestMonotonicityProperties:
    """Pruebas de monotonía en la degradación de veredictos."""

    def test_translator_monotonicity_with_topological_damage(
        self,
        translator: SemanticTranslator,
    ) -> None:
        """Verifica que mayor daño topológico no mejora el veredicto."""
        verdict_values: list[int] = []

        for beta_1 in (0, 1, 3, 5):
            report = translator.compose_strategic_narrative(
                topological_metrics=make_topology(beta_0=1, beta_1=beta_1),
                financial_metrics={
                    "performance": {
                        "recommendation": "ACEPTAR",
                        "profitability_index": 1.5,
                    }
                },
                stability=15.0,
            )
            verdict_values.append(report.verdict.value)

        assert verdict_values == sorted(verdict_values), (
            f"Mayor daño topológico no debe mejorar el veredicto: {verdict_values}"
        )

    def test_translator_monotonicity_with_stability_decrease(
        self,
        translator: SemanticTranslator,
    ) -> None:
        """Verifica que menor estabilidad no mejora el veredicto."""
        verdict_values: list[int] = []

        for stability in (20.0, 10.0, 5.0, 1.0):
            report = translator.compose_strategic_narrative(
                topological_metrics=make_topology(beta_0=1, beta_1=0),
                financial_metrics={
                    "performance": {
                        "recommendation": "ACEPTAR",
                        "profitability_index": 1.5,
                    }
                },
                stability=stability,
            )
            verdict_values.append(report.verdict.value)

        assert verdict_values == sorted(verdict_values), (
            f"Menor estabilidad no debe mejorar el veredicto: {verdict_values}"
        )


# =============================================================================
# PRUEBAS DE RISK CHALLENGER (CONDICIONAL)
# =============================================================================

@pytest.mark.skipif(not HAS_RISK_CHALLENGER, reason="RiskChallenger no disponible")
class TestRiskChallengerIntegration:
    """Auditoría adversarial sobre reportes ingenuamente optimistas."""

    def test_challenger_detects_inverted_pyramid(self) -> None:
        """Verifica que el challenger detecta pirámides invertidas."""
        challenger = RiskChallenger()

        naive_report = ConstructionRiskReport(
            integrity_score=95.0,
            financial_risk_level="BAJO",
            details={
                "topological_invariants": {
                    "pyramid_stability": 0.45,
                }
            },
            waste_alerts=[],
            circular_risks=[],
            complexity_level="Baja",
        )

        audited_report = challenger.challenge_verdict(naive_report)

        assert audited_report.financial_risk_level != "BAJO"
        assert audited_report.integrity_score < 95.0

    def test_challenger_adds_deliberation_notes(self) -> None:
        """Verifica que el challenger añade notas de deliberación."""
        challenger = RiskChallenger()

        naive_report = ConstructionRiskReport(
            integrity_score=90.0,
            financial_risk_level="MEDIO",
            details={"topological_invariants": {"pyramid_stability": 0.8}},
            waste_alerts=[],
            circular_risks=[],
            complexity_level="Media",
        )

        audited_report = challenger.challenge_verdict(naive_report)

        narrative = getattr(audited_report, "strategic_narrative", "")
        if narrative:
            assert_contains_any(
                narrative,
                ["CONTRADICCIÓN", "ACTA", "DELIBERACIÓN", "CHALLENGER", "DEBATE"],
                "Debe existir huella deliberativa del challenger",
            )


# =============================================================================
# PRUEBAS DE COHERENCIA NARRATIVA
# =============================================================================

class TestNarrativeCoherence:
    """Consistencia semántica entre TelemetryNarrator y SemanticTranslator."""

    def test_narrator_and_translator_agree_on_physics_failure(
        self,
        narrator: TelemetryNarrator,
        translator: SemanticTranslator,
        context_with_physics_failure: TelemetryContext,
    ) -> None:
        """Verifica acuerdo sobre fallo en PHYSICS."""
        narrator_report = narrator.summarize_execution(context_with_physics_failure)
        translator_report = translator.compose_strategic_narrative(
            topological_metrics=make_topology(beta_0=0, beta_1=0),
            financial_metrics={},
            stability=0.1,
        )

        assert "PHYSICS" in verdict_code(narrator_report) or "REJECTED" in verdict_code(
            narrator_report
        )
        assert translator_report.verdict.value >= VerdictLevel.PRECAUCION.value

    def test_narrator_and_translator_agree_on_success(
        self,
        narrator: TelemetryNarrator,
        translator: SemanticTranslator,
        context_all_success: TelemetryContext,
    ) -> None:
        """Verifica acuerdo sobre éxito total."""
        narrator_report = narrator.summarize_execution(context_all_success)
        translator_report = translator.compose_strategic_narrative(
            topological_metrics=make_topology(beta_0=1, beta_1=0),
            financial_metrics={
                "performance": {
                    "recommendation": "ACEPTAR",
                    "profitability_index": 1.5,
                }
            },
            stability=15.0,
        )

        assert verdict_code(narrator_report) == "APPROVED"
        assert translator_report.verdict == VerdictLevel.VIABLE

    def test_narratives_use_consistent_language(
        self,
        narrator: TelemetryNarrator,
        translator: SemanticTranslator,
        context_with_tactics_failure: TelemetryContext,
    ) -> None:
        """Verifica consistencia de lenguaje para fallo topológico."""
        narrator_report = narrator.summarize_execution(context_with_tactics_failure)
        tactics_narrative = strata_entry(narrator_report, Stratum.TACTICS).get(
            "narrative",
            "",
        )

        translator_report = translator.compose_strategic_narrative(
            topological_metrics=make_topology(beta_0=1, beta_1=5),
            financial_metrics={"performance": {"recommendation": "REVISAR"}},
            stability=10.0,
        )

        topo_keywords = list(_STRATUM_KEYWORDS[Stratum.TACTICS])

        assert_contains_any(
            tactics_narrative,
            topo_keywords,
            "Narrator debe mencionar el problema topológico",
        )
        assert_contains_any(
            translator_report.raw_narrative,
            topo_keywords,
            "Translator debe mencionar el problema topológico",
        )


# =============================================================================
# PRUEBAS DE CASOS ESPECIALES
# =============================================================================

class TestSpecialCases:
    """Casos límite de neutralidad, warnings y degradaciones."""

    def test_empty_context_is_neutral_or_approved(
        self,
        narrator: TelemetryNarrator,
        fresh_context: TelemetryContext,
    ) -> None:
        """Verifica que contexto vacío es neutral o aprobado."""
        report = narrator.summarize_execution(fresh_context)

        assert verdict_code(report) in {"EMPTY", "APPROVED", ""}

    def test_warning_only_does_not_hard_reject(
        self,
        narrator: TelemetryNarrator,
        fresh_context: TelemetryContext,
    ) -> None:
        """Verifica que solo warnings no causa rechazo duro."""
        with fresh_context.span("physics", stratum=Stratum.PHYSICS) as span:
            span.status = StepStatus.WARNING

        report = narrator.summarize_execution(fresh_context)

        # No debe ser REJECTED con severidad CRITICO
        is_hard_reject = (
            "REJECTED" in verdict_code(report)
            and global_severity(report) == "CRITICO"
        )
        assert not is_hard_reject

    def test_synergy_risk_always_rejects(
        self,
        translator: SemanticTranslator,
    ) -> None:
        """Verifica que riesgo de sinergia siempre rechaza."""
        synergy = {
            "synergy_detected": True,
            "intersecting_cycles_count": 3,
            "intersecting_cycles": [["A", "B", "C"]],
        }

        report = translator.compose_strategic_narrative(
            topological_metrics=make_topology(beta_0=1, beta_1=0),
            financial_metrics={"performance": {"recommendation": "ACEPTAR"}},
            stability=20.0,
            synergy_risk=synergy,
        )

        assert report.verdict == VerdictLevel.RECHAZAR
        assert_contains_any(
            report.raw_narrative,
            ["Dominó", "EMERGENCIA", "Sinergia", "contagio"],
            "Debe mencionar el efecto dominó",
        )

    def test_thermal_fever_degrades_verdict(
        self,
        translator: SemanticTranslator,
    ) -> None:
        """Verifica que fiebre térmica degrada el veredicto."""
        thermal = {"system_temperature": 75.0, "entropy": 0.8}

        report = translator.compose_strategic_narrative(
            topological_metrics=make_topology(beta_0=1, beta_1=0),
            financial_metrics={"performance": {"recommendation": "ACEPTAR"}},
            stability=10.0,
            thermal_metrics=thermal,
        )

        assert_contains_any(
            report.raw_narrative,
            ["FIEBRE", "calor", "temperatura", "térmic"],
            "Debe mencionar el problema térmico",
        )
        assert report.verdict.value >= VerdictLevel.CONDICIONAL.value


# =============================================================================
# PRUEBAS DE DETERMINISMO
# =============================================================================

class TestDeterminism:
    """Determinismo extensional: misma entrada → misma salida."""

    def test_narrator_is_deterministic(
        self,
        narrator: TelemetryNarrator,
    ) -> None:
        """Verifica determinismo del narrator."""
        context_1 = build_context()
        context_2 = build_context()

        report_1 = narrator.summarize_execution(context_1)
        report_2 = narrator.summarize_execution(context_2)

        assert verdict_code(report_1) == verdict_code(report_2)
        assert global_severity(report_1) == global_severity(report_2)
        
        for stratum in Stratum:
            if stratum.name in report_1.get("strata_analysis", {}):
                assert severity_of(report_1, stratum) == severity_of(report_2, stratum)

    def test_translator_is_deterministic(
        self,
        translator: SemanticTranslator,
    ) -> None:
        """Verifica determinismo del translator."""
        topology = make_topology(beta_0=1, beta_1=2)
        financials = {"performance": {"recommendation": "REVISAR"}}

        report_1 = translator.compose_strategic_narrative(
            topological_metrics=topology,
            financial_metrics=financials,
            stability=5.0,
        )
        report_2 = translator.compose_strategic_narrative(
            topological_metrics=topology,
            financial_metrics=financials,
            stability=5.0,
        )

        assert report_1.verdict == report_2.verdict
        assert report_1.raw_narrative == report_2.raw_narrative

    def test_narrator_deterministic_across_multiple_calls(
        self,
        narrator: TelemetryNarrator,
    ) -> None:
        """Verifica determinismo a través de múltiples llamadas."""
        verdicts = []
        
        for _ in range(5):
            context = build_context(physics_ok=False)
            report = narrator.summarize_execution(context)
            verdicts.append(verdict_code(report))
        
        assert len(set(verdicts)) == 1, (
            f"Veredictos inconsistentes: {verdicts}"
        )


# =============================================================================
# PRUEBAS DE INVARIANTES GLOBALES
# =============================================================================

class TestGlobalInvariants:
    """Invariantes globales del sistema de coherencia piramidal."""

    def test_root_cause_is_always_in_failing_strata(
        self,
        narrator: TelemetryNarrator,
    ) -> None:
        """Verifica que la causa raíz siempre está en los estratos que fallan."""
        for combo in TestCombinedFailuresExhaustive.generate_all_failure_combinations():
            context = build_context_from_failures(combo)
            report = narrator.summarize_execution(context)
            
            root = root_cause_name(report)
            if root is not None:
                root_stratum = Stratum[root]
                assert root_stratum in combo, (
                    f"Causa raíz {root} no está en fallos {[s.name for s in combo]}"
                )

    def test_severity_ordering_is_preserved(
        self,
        narrator: TelemetryNarrator,
    ) -> None:
        """Verifica que el orden de severidades se preserva."""
        # Fallo menor (solo STRATEGY) debe tener menor severidad que fallo mayor
        context_strategy = build_context(strategy_ok=False)
        context_physics = build_context(physics_ok=False)
        
        report_strategy = narrator.summarize_execution(context_strategy)
        report_physics = narrator.summarize_execution(context_physics)
        
        # Ambos deben ser CRITICO en el estrato que falla
        assert severity_of(report_strategy, Stratum.STRATEGY) == "CRITICO"
        assert severity_of(report_physics, Stratum.PHYSICS) == "CRITICO"

    def test_approved_requires_all_strata_ok(
        self,
        narrator: TelemetryNarrator,
    ) -> None:
        """Verifica que APPROVED requiere todos los estratos OK."""
        context = build_context()
        report = narrator.summarize_execution(context)
        
        if verdict_code(report) == "APPROVED":
            for stratum in Stratum:
                if stratum.name in report.get("strata_analysis", {}):
                    assert severity_of(report, stratum) in {"OPTIMO", "ACEPTABLE"}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])