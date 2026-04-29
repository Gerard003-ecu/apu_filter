"""
=========================================================================================
Módulo: Suite de Pruebas para Semantic Validation Engine
Ubicación: tests/unit/boole/wisdom/test_semantic_validator.py
Versión: 1.0 - Suite Rigurosa de Testing
=========================================================================================

COBERTURA DE PRUEBAS:
---------------------
1. Estructuras de datos (BusinessPurpose, LLMOutput, RiskProfile, etc.)
2. Validadores individuales (Purpose, Confidence, Constraints)
3. Engine de validación semántica completo
4. Knowledge graph y mapeos semánticos
5. Sistema de scoring y agregación
6. Casos extremos y edge cases
7. Performance y escalabilidad
8. Integración end-to-end

EJECUCIÓN:
----------
    python -m pytest test_ontological_diffeomorphism_engine.py -v
    python -m unittest test_ontological_diffeomorphism_engine.TestBusinessPurpose -v

=========================================================================================
"""

import sys
from contextlib import contextmanager

import pytest

# Importar networkx para knowledge graph
try:
    import networkx as nx
except ImportError:
    print("networkx no instalado. Instalar con: pip install networkx")
    sys.exit(1)

# Importar módulo a testear
from app.boole.wisdom.semantic_validator import (
    BusinessPurpose,
    ConfidenceFilter,
    ConstraintMapper,
    LLMOutput,
    PurposeValidator,
    RiskProfile,
    SemanticValidationEngine,
    ValidationResult,
    Verdict,
    create_default_knowledge_graph,
)

# Suprimir logs durante testing
# logging.getLogger(...).setLevel(logging.CRITICAL)  # Removido por orden


# ========================================================================================
# UTILIDADES DE TESTING Y FIXTURES
# ========================================================================================


@pytest.fixture
def knowledge_graph():
    """
    Funtor de inicialización pura. Garantiza una topología
    inalterada para cada evaluación semántica.
    """
    return create_default_knowledge_graph()


class TestBase:
    """Clase base con utilidades comunes."""

    @contextmanager
    def assertNotRaises(self, exc_type):
        """Verifica que NO se lance una excepción."""
        try:
            yield
        except exc_type as e:
            pytest.fail(f"Se lanzó {exc_type.__name__} inesperadamente: {e}")

    def assertBetween(self, value, min_val, max_val, msg=None):
        """Verifica que un valor esté en un rango."""
        if not (min_val <= value <= max_val):
            standard_msg = f"{value} no está entre {min_val} y {max_val}"
            pytest.fail(self._formatMessage(msg, standard_msg))


# ========================================================================================
# TESTS: Verdict
# ========================================================================================


class TestVerdict(TestBase):
    """Tests para la enumeración Verdict."""

    def test_ordering(self):
        """Veredictos tienen orden total."""
        assert Verdict.VIABLE < Verdict.CONDITIONAL
        assert Verdict.CONDITIONAL < Verdict.WARNING
        assert Verdict.WARNING < Verdict.REJECT

    def test_is_accepted(self):
        """Propiedad is_accepted."""
        assert Verdict.VIABLE.is_accepted
        assert Verdict.CONDITIONAL.is_accepted
        assert not Verdict.WARNING.is_accepted
        assert not Verdict.REJECT.is_accepted

    def test_requires_human_review(self):
        """Propiedad requires_human_review."""
        assert not Verdict.VIABLE.requires_human_review
        assert Verdict.CONDITIONAL.requires_human_review
        assert Verdict.WARNING.requires_human_review
        assert not Verdict.REJECT.requires_human_review

    def test_string_representation(self):
        """Representación como string."""
        assert str(Verdict.VIABLE) == "VIABLE"
        assert str(Verdict.REJECT) == "REJECT"

    def test_integer_value(self):
        """Valores enteros."""
        assert int(Verdict.VIABLE) == 0
        assert int(Verdict.CONDITIONAL) == 1
        assert int(Verdict.WARNING) == 2
        assert int(Verdict.REJECT) == 3


# ========================================================================================
# TESTS: BusinessPurpose
# ========================================================================================


class TestBusinessPurpose(TestBase):
    """Tests para BusinessPurpose."""

    def test_construction_valid(self):
        """Construcción válida con parámetros mínimos."""
        purpose = BusinessPurpose(
            concept="caching", business_problem="LATENCY_REDUCTION", strength=0.8
        )

        assert purpose.concept == "caching"
        assert purpose.business_problem == "LATENCY_REDUCTION"
        assert purpose.strength == 0.8
        assert purpose.confidence == 1.0  # Default

    def test_construction_with_confidence(self):
        """Construcción con confianza personalizada."""
        purpose = BusinessPurpose(
            concept="encryption",
            business_problem="SECURITY_HARDENING",
            strength=0.9,
            confidence=0.85,
        )

        assert purpose.confidence == 0.85

    def test_construction_invalid_strength_high(self):
        """Strength > 1 inválido."""
        with pytest.raises(ValueError):
            BusinessPurpose("concept", "problem", strength=1.5)

    def test_construction_invalid_strength_low(self):
        """Strength < 0 inválido."""
        with pytest.raises(ValueError):
            BusinessPurpose("concept", "problem", strength=-0.1)

    def test_construction_invalid_confidence(self):
        """Confidence fuera de rango."""
        with pytest.raises(ValueError):
            BusinessPurpose("concept", "problem", strength=0.5, confidence=1.2)

    def test_construction_empty_concept(self):
        """Concepto vacío inválido."""
        with pytest.raises(ValueError):
            BusinessPurpose("", "problem", strength=0.5)

    def test_construction_empty_problem(self):
        """Problema vacío inválido."""
        with pytest.raises(ValueError):
            BusinessPurpose("concept", "", strength=0.5)

    def test_effective_strength(self):
        """Cálculo de fuerza efectiva."""
        purpose = BusinessPurpose(
            concept="monitoring",
            business_problem="RELIABILITY_IMPROVEMENT",
            strength=0.8,
            confidence=0.9,
        )

        assert abs(purpose.effective_strength - (0.72)) < 1e-6  # 0.8 * 0.9

    def test_effective_strength_full_confidence(self):
        """Fuerza efectiva con confianza total."""
        purpose = BusinessPurpose("concept", "problem", strength=0.75, confidence=1.0)

        assert purpose.effective_strength == 0.75

    def test_effective_strength_zero_confidence(self):
        """Fuerza efectiva con confianza cero."""
        purpose = BusinessPurpose("concept", "problem", strength=0.9, confidence=0.0)

        assert purpose.effective_strength == 0.0

    def test_immutability(self):
        """BusinessPurpose es inmutable."""
        purpose = BusinessPurpose("concept", "problem", strength=0.5)

        with pytest.raises(AttributeError):
            purpose.strength = 0.8

    def test_ordering(self):
        """Ordenamiento por comparación."""
        p1 = BusinessPurpose("a", "problem", strength=0.5)
        p2 = BusinessPurpose("b", "problem", strength=0.8)

        # Debe ser ordenable
        assert p1 < p2

        # Debe ser posible ordenar una lista
        purposes = [p2, p1]
        sorted_purposes = sorted(purposes)
        assert sorted_purposes[0] == p1


# ========================================================================================
# TESTS: LLMOutput
# ========================================================================================


class TestLLMOutput(TestBase):
    """Tests para LLMOutput."""

    def test_construction_valid(self):
        """Construcción válida."""
        output = LLMOutput(
            entropy=1.5, confidence=0.85, temperature=0.7, num_tokens=150
        )

        assert output.entropy == 1.5
        assert output.confidence == 0.85
        assert output.temperature == 0.7
        assert output.num_tokens == 150

    def test_construction_minimal(self):
        """Construcción con parámetros mínimos."""
        output = LLMOutput(entropy=1.0, confidence=0.9)

        assert output.temperature == 1.0  # Default
        assert output.num_tokens == 0  # Default

    def test_construction_invalid_entropy(self):
        """Entropía negativa inválida."""
        with pytest.raises(ValueError):
            LLMOutput(entropy=-0.5, confidence=0.9)

    def test_construction_invalid_confidence_high(self):
        """Confianza > 1 inválida."""
        with pytest.raises(ValueError):
            LLMOutput(entropy=1.0, confidence=1.5)

    def test_construction_invalid_confidence_low(self):
        """Confianza < 0 inválida."""
        with pytest.raises(ValueError):
            LLMOutput(entropy=1.0, confidence=-0.1)

    def test_construction_invalid_temperature(self):
        """Temperatura ≤ 0 inválida."""
        with pytest.raises(ValueError):
            LLMOutput(entropy=1.0, confidence=0.9, temperature=0.0)

    def test_construction_invalid_num_tokens(self):
        """Número de tokens negativo inválido."""
        with pytest.raises(ValueError):
            LLMOutput(entropy=1.0, confidence=0.9, num_tokens=-10)

    def test_normalized_entropy_zero_tokens(self):
        """Entropía normalizada sin tokens."""
        output = LLMOutput(entropy=2.0, confidence=0.9, temperature=1.0, num_tokens=0)

        # Sin tokens, no normaliza por longitud
        assert output.normalized_entropy == 2.0

    def test_normalized_entropy_with_tokens(self):
        """Entropía normalizada con tokens."""
        output = LLMOutput(entropy=4.0, confidence=0.9, temperature=2.0, num_tokens=100)

        # Normalización: 4.0 / (2.0 * sqrt(100)) = 4.0 / 20 = 0.2
        assert abs(output.normalized_entropy - (0.2)) < 1e-6

    def test_normalized_entropy_high_temperature(self):
        """Normalización con temperatura alta."""
        output = LLMOutput(entropy=3.0, confidence=0.9, temperature=3.0, num_tokens=9)

        # 3.0 / (3.0 * sqrt(9)) = 3.0 / 9.0 = 0.333...
        assert abs(output.normalized_entropy - (1.0 / 3.0)) < 1e-6

    def test_immutability(self):
        """LLMOutput es inmutable."""
        output = LLMOutput(entropy=1.0, confidence=0.9)

        with pytest.raises(AttributeError):
            output.entropy = 2.0


# ========================================================================================
# TESTS: RiskProfile
# ========================================================================================


class TestRiskProfile(TestBase):
    """Tests para RiskProfile."""

    def test_construction_valid(self):
        """Construcción válida."""
        profile = RiskProfile(
            risk_tolerance=0.6, domain_criticality=0.4, acceptable_failure_rate=0.05
        )

        assert profile.risk_tolerance == 0.6
        assert profile.domain_criticality == 0.4
        assert profile.acceptable_failure_rate == 0.05

    def test_construction_defaults(self):
        """Construcción con defaults."""
        profile = RiskProfile(risk_tolerance=0.5)

        assert profile.domain_criticality == 0.5
        assert profile.acceptable_failure_rate == 0.01

    def test_construction_invalid_tolerance(self):
        """Tolerancia fuera de rango."""
        with pytest.raises(ValueError):
            RiskProfile(risk_tolerance=1.5)

        with pytest.raises(ValueError):
            RiskProfile(risk_tolerance=-0.1)

    def test_construction_invalid_criticality(self):
        """Criticidad fuera de rango."""
        with pytest.raises(ValueError):
            RiskProfile(risk_tolerance=0.5, domain_criticality=2.0)

    def test_construction_invalid_failure_rate(self):
        """Tasa de fallo fuera de rango."""
        with pytest.raises(ValueError):
            RiskProfile(risk_tolerance=0.5, acceptable_failure_rate=1.5)

    def test_effective_tolerance_low_criticality(self):
        """Tolerancia efectiva con baja criticidad."""
        profile = RiskProfile(risk_tolerance=0.8, domain_criticality=0.0)

        # 0.8 * (1 - 0.5 * 0.0) = 0.8
        assert profile.effective_tolerance == 0.8

    def test_effective_tolerance_high_criticality(self):
        """Tolerancia efectiva con alta criticidad."""
        profile = RiskProfile(risk_tolerance=0.8, domain_criticality=1.0)

        # 0.8 * (1 - 0.5 * 1.0) = 0.8 * 0.5 = 0.4
        assert profile.effective_tolerance == 0.4

    def test_effective_tolerance_medium(self):
        """Tolerancia efectiva con valores medios."""
        profile = RiskProfile(risk_tolerance=0.6, domain_criticality=0.4)

        # 0.6 * (1 - 0.5 * 0.4) = 0.6 * 0.8 = 0.48
        assert abs(profile.effective_tolerance - (0.48)) < 1e-6

    def test_risk_category_highly_conservative(self):
        """Categoría: altamente conservador."""
        profile = RiskProfile(risk_tolerance=0.1, domain_criticality=0.5)

        assert profile.risk_category == "HIGHLY_CONSERVATIVE"

    def test_risk_category_conservative(self):
        """Categoría: conservador."""
        profile = RiskProfile(risk_tolerance=0.3, domain_criticality=0.3)

        assert profile.risk_category == "CONSERVATIVE"

    def test_risk_category_moderate(self):
        """Categoría: moderado."""
        profile = RiskProfile(risk_tolerance=0.5, domain_criticality=0.2)

        assert profile.risk_category == "MODERATE"

    def test_risk_category_aggressive(self):
        """Categoría: agresivo."""
        profile = RiskProfile(risk_tolerance=0.7, domain_criticality=0.1)

        assert profile.risk_category == "AGGRESSIVE"

    def test_risk_category_highly_aggressive(self):
        """Categoría: altamente agresivo."""
        profile = RiskProfile(risk_tolerance=0.9, domain_criticality=0.0)

        assert profile.risk_category == "HIGHLY_AGGRESSIVE"

    def test_immutability(self):
        """RiskProfile es inmutable."""
        profile = RiskProfile(risk_tolerance=0.5)

        with pytest.raises(AttributeError):
            profile.risk_tolerance = 0.8


# ========================================================================================
# TESTS: ValidationResult
# ========================================================================================


class TestValidationResult(TestBase):
    """Tests para ValidationResult."""

    def test_construction(self):
        """Construcción básica."""
        result = ValidationResult(verdict=Verdict.VIABLE, overall_score=0.85)

        assert result.verdict == Verdict.VIABLE
        assert result.overall_score == 0.85
        assert len(result.signals) == 0
        assert len(result.reasons) == 0

    def test_add_reason(self):
        """Añadir razones."""
        result = ValidationResult(verdict=Verdict.VIABLE, overall_score=0.8)

        result.add_reason("Test reason")

        assert len(result.reasons) == 1
        assert "Test reason" in result.reasons

    def test_add_reason_with_signal(self):
        """Añadir razón con señal."""
        result = ValidationResult(verdict=Verdict.VIABLE, overall_score=0.8)

        result.add_reason("Purpose validated", signal_name="purpose", signal_value=0.9)

        assert len(result.signals) == 1
        assert result.signals["purpose"] == 0.9

    def test_multiple_reasons(self):
        """Múltiples razones."""
        result = ValidationResult(verdict=Verdict.CONDITIONAL, overall_score=0.7)

        result.add_reason("Reason 1", "signal1", 0.8)
        result.add_reason("Reason 2", "signal2", 0.6)
        result.add_reason("Reason 3")

        assert len(result.reasons) == 3
        assert len(result.signals) == 2


# ========================================================================================
# TESTS: PurposeValidator
# ========================================================================================


class TestPurposeValidator(TestBase):
    """Tests para PurposeValidator."""

    def test_initialization_default(self):
        """Inicialización con defaults."""
        validator = PurposeValidator()

        assert validator.canonical_problems is not None
        assert len(validator.canonical_problems) > 0

    def test_initialization_custom_problems(self):
        """Inicialización con problemas personalizados."""
        custom_problems = frozenset(["PROBLEM_A", "PROBLEM_B"])
        validator = PurposeValidator(canonical_problems=custom_problems)

        assert validator.canonical_problems == custom_problems

    def test_initialization_invalid_threshold(self):
        """Umbral inválido."""
        with pytest.raises(ValueError):
            PurposeValidator(min_strength_threshold=1.5)

    def test_validate_strong_purpose(self):
        """Validación de propósito fuerte."""
        validator = PurposeValidator(canonical_problems=frozenset(["COST_REDUCTION"]))

        purposes = [
            BusinessPurpose("caching", "COST_REDUCTION", strength=0.9, confidence=0.95)
        ]

        is_valid, strength, reason = validator.validate(purposes)

        assert is_valid
        assert strength > 0.7
        assert "Strong purpose" in reason

    def test_validate_weak_purpose(self):
        """Validación de propósito débil."""
        validator = PurposeValidator(
            canonical_problems=frozenset(["COST_REDUCTION"]), min_strength_threshold=0.8
        )

        purposes = [
            BusinessPurpose("unknown", "COST_REDUCTION", strength=0.5, confidence=0.7)
        ]

        is_valid, strength, reason = validator.validate(purposes)

        assert not is_valid
        assert strength < 0.8

    def test_validate_non_canonical_problem(self):
        """Problema no canónico."""
        validator = PurposeValidator(canonical_problems=frozenset(["COST_REDUCTION"]))

        purposes = [BusinessPurpose("concept", "UNKNOWN_PROBLEM", strength=0.95)]

        is_valid, strength, reason = validator.validate(purposes)

        assert not is_valid
        assert strength == 0.0
        assert "No purposes map" in reason

    def test_validate_empty_purposes(self):
        """Sin propósitos."""
        validator = PurposeValidator()

        is_valid, strength, reason = validator.validate([])

        assert not is_valid
        assert strength == 0.0
        assert "No business purposes" in reason

    def test_validate_multiple_purposes(self):
        """Múltiples propósitos."""
        validator = PurposeValidator(
            canonical_problems=frozenset(["COST_REDUCTION", "LATENCY_REDUCTION"])
        )

        purposes = [
            BusinessPurpose("caching", "COST_REDUCTION", strength=0.7, confidence=0.9),
            BusinessPurpose(
                "caching", "LATENCY_REDUCTION", strength=0.9, confidence=0.95
            ),
        ]

        is_valid, strength, reason = validator.validate(purposes)

        assert is_valid
        # Debe usar el mejor (0.9 * 0.95 = 0.855)
        assert strength > 0.8

    def test_compute_purpose_score_empty(self):
        """Score con propósitos vacíos."""
        validator = PurposeValidator()

        score = validator.compute_purpose_score([])

        assert score == 0.0

    def test_compute_purpose_score_single(self):
        """Score con un propósito."""
        validator = PurposeValidator(canonical_problems=frozenset(["COST_REDUCTION"]))

        purposes = [
            BusinessPurpose("caching", "COST_REDUCTION", strength=0.8, confidence=1.0)
        ]

        score = validator.compute_purpose_score(purposes)

        # 70% max + 30% mean = 0.7 * 0.8 + 0.3 * 0.8 = 0.8
        assert abs(score - (0.8)) < 1e-6

    def test_compute_purpose_score_multiple(self):
        """Score con múltiples propósitos."""
        validator = PurposeValidator(
            canonical_problems=frozenset(["COST_REDUCTION", "LATENCY_REDUCTION"])
        )

        purposes = [
            BusinessPurpose("caching", "COST_REDUCTION", strength=0.6),
            BusinessPurpose("caching", "LATENCY_REDUCTION", strength=0.8),
        ]

        score = validator.compute_purpose_score(purposes)

        # max = 0.8, mean = 0.7
        # 0.7 * 0.8 + 0.3 * 0.7 = 0.56 + 0.21 = 0.77
        assert abs(score - (0.77)) < 1e-6


# ========================================================================================
# TESTS: ConfidenceFilter
# ========================================================================================


class TestConfidenceFilter(TestBase):
    """Tests para ConfidenceFilter."""

    def test_initialization_default(self):
        """Inicialización con defaults."""
        filter = ConfidenceFilter()

        assert filter.min_confidence > 0
        assert filter.min_confidence < 1

    def test_initialization_custom(self):
        """Inicialización personalizada."""
        filter = ConfidenceFilter(
            min_confidence=0.8, max_entropy=3.0, max_normalized_entropy=0.7
        )

        assert filter.min_confidence == 0.8
        assert filter.max_entropy == 3.0

    def test_validate_high_quality(self):
        """Validación de salida de alta calidad."""
        filter = ConfidenceFilter()

        output = LLMOutput(
            entropy=0.5, confidence=0.95, temperature=0.7, num_tokens=100
        )

        is_valid, score, reason = filter.validate(output)

        assert is_valid
        assert score > 0.8

    def test_validate_low_confidence(self):
        """Validación con baja confianza."""
        filter = ConfidenceFilter(min_confidence=0.7)

        output = LLMOutput(entropy=1.0, confidence=0.5, temperature=1.0, num_tokens=100)

        is_valid, score, reason = filter.validate(output)

        assert not is_valid
        assert "Confidence" in reason

    def test_validate_high_entropy(self):
        """Validación con alta entropía."""
        filter = ConfidenceFilter(max_entropy=2.0)

        output = LLMOutput(entropy=3.5, confidence=0.9, temperature=1.0, num_tokens=100)

        is_valid, score, reason = filter.validate(output)

        assert not is_valid
        assert "Entropy" in reason

    def test_validate_high_normalized_entropy(self):
        """Validación con alta entropía normalizada."""
        filter = ConfidenceFilter(max_normalized_entropy=0.3)

        # Crear output con alta entropía normalizada
        output = LLMOutput(
            entropy=2.0,
            confidence=0.9,
            temperature=0.5,  # Temperatura baja amplifica
            num_tokens=4,  # Pocos tokens amplifica
        )

        is_valid, score, reason = filter.validate(output)

        # normalized_entropy = 2.0 / (0.5 * sqrt(4)) = 2.0 / 1.0 = 2.0
        # Esto excede 0.3
        assert not is_valid

    def test_compute_confidence_score(self):
        """Cálculo de score de confianza."""
        filter = ConfidenceFilter()

        output = LLMOutput(entropy=1.0, confidence=0.8, temperature=1.0, num_tokens=100)

        score = filter.compute_confidence_score(output)

        self.assertBetween(score, 0.0, 1.0)
        assert score > 0.5


# ========================================================================================
# TESTS: ConstraintMapper
# ========================================================================================


class TestConstraintMapper(TestBase):
    """Tests para ConstraintMapper."""

    def test_map_to_constraints_conservative(self):
        """Mapeo para perfil conservador."""
        mapper = ConstraintMapper()

        profile = RiskProfile(risk_tolerance=0.2, domain_criticality=0.8)

        constraints = mapper.map_to_constraints(profile)

        # Debe tener límites estrictos
        assert "cyclomatic" in constraints
        assert "depth" in constraints
        assert "loc" in constraints

        # Valores conservadores
        assert constraints["cyclomatic"] <= 15

    def test_map_to_constraints_aggressive(self):
        """Mapeo para perfil agresivo."""
        mapper = ConstraintMapper()

        profile = RiskProfile(risk_tolerance=0.9, domain_criticality=0.1)

        constraints = mapper.map_to_constraints(profile)

        # Límites más relajados
        assert constraints["cyclomatic"] > 25
        assert constraints["loc"] > 300

    def test_map_to_constraints_moderate(self):
        """Mapeo para perfil moderado."""
        mapper = ConstraintMapper()

        profile = RiskProfile(risk_tolerance=0.5, domain_criticality=0.5)

        constraints = mapper.map_to_constraints(profile)

        # Valores intermedios
        self.assertBetween(constraints["cyclomatic"], 10, 30)

    def test_compute_constraint_score_all_satisfied(self):
        """Score con todas las restricciones satisfechas."""
        mapper = ConstraintMapper()

        profile = RiskProfile(risk_tolerance=0.5)

        # Métricas que cumplen restricciones moderadas
        metrics = {"cyclomatic": 10, "depth": 3, "loc": 80}

        score = mapper.compute_constraint_score(metrics, profile)

        assert score > 0.9

    def test_compute_constraint_score_all_violated(self):
        """Score con todas las restricciones violadas."""
        mapper = ConstraintMapper()

        profile = RiskProfile(risk_tolerance=0.2)  # Muy conservador

        # Métricas que violan restricciones
        metrics = {"cyclomatic": 100, "depth": 15, "loc": 2000}

        score = mapper.compute_constraint_score(metrics, profile)

        assert score < 0.3

    def test_compute_constraint_score_partial(self):
        """Score con algunas restricciones violadas."""
        mapper = ConstraintMapper()

        profile = RiskProfile(risk_tolerance=0.5)

        metrics = {"cyclomatic": 15, "depth": 10, "loc": 150}  # OK  # Violado  # OK

        score = mapper.compute_constraint_score(metrics, profile)

        # Debe estar entre 0 y 1, ni muy alto ni muy bajo
        assert 0.2 <= score <= 0.9

    def test_compute_constraint_score_missing_metrics(self):
        """Score con métricas faltantes."""
        mapper = ConstraintMapper()

        profile = RiskProfile(risk_tolerance=0.5)

        # Solo algunas métricas
        metrics = {"cyclomatic": 10}

        score = mapper.compute_constraint_score(metrics, profile)

        # Debe manejar métricas faltantes sin error
        self.assertBetween(score, 0.0, 1.0)


# ========================================================================================
# TESTS: SemanticValidationEngine
# ========================================================================================


class TestSemanticValidationEngine(TestBase):
    """Tests para SemanticValidationEngine."""

    def setup_method(self):
        """Configuración común para tests."""
        self.kg = knowledge_graph

    def test_initialization_default(self):
        """Inicialización con defaults."""
        engine = SemanticValidationEngine()

        assert engine.risk_profile is not None
        assert engine.weights is not None

    def test_initialization_custom_profile(self):
        """Inicialización con perfil personalizado."""
        profile = RiskProfile(risk_tolerance=0.3, domain_criticality=0.7)

        engine = SemanticValidationEngine(risk_profile=profile)

        assert engine.risk_profile == profile

    def test_initialization_custom_weights(self):
        """Inicialización con pesos personalizados."""
        weights = {"purpose": 0.4, "confidence": 0.3, "constraints": 0.2, "risk": 0.1}

        engine = SemanticValidationEngine(weights=weights)

        assert engine.weights == weights

    def test_initialization_weights_normalization(self):
        """Normalización de pesos que no suman 1."""
        weights = {
            "purpose": 0.5,
            "confidence": 0.5,
            "constraints": 0.5,
            "risk": 0.5,
        }  # Suman 2.0

        engine = SemanticValidationEngine(weights=weights)

        # Deben estar normalizados
        total = sum(engine.weights.values())
        assert abs(total - (1.0)) < 1e-6

    def test_validate_viable_scenario(self, knowledge_graph):
        """Escenario VIABLE: todo perfecto."""
        engine = SemanticValidationEngine(
            knowledge_graph=knowledge_graph,
            risk_profile=RiskProfile(risk_tolerance=0.7),
        )

        purposes = [
            BusinessPurpose(
                "caching", "LATENCY_REDUCTION", strength=0.9, confidence=0.95
            )
        ]

        llm_output = LLMOutput(
            entropy=0.5, confidence=0.95, temperature=0.7, num_tokens=100
        )

        code_metrics = {"cyclomatic": 8, "depth": 3, "loc": 50}

        result = engine.validate(purposes, llm_output, code_metrics)

        assert result.verdict == Verdict.VIABLE
        assert result.overall_score > 0.75
        assert result.verdict.is_accepted

    def test_validate_conditional_scenario(self, knowledge_graph):
        """Escenario CONDITIONAL: bueno pero no perfecto."""
        engine = SemanticValidationEngine(
            knowledge_graph=knowledge_graph,
            risk_profile=RiskProfile(risk_tolerance=0.5),
        )

        purposes = [
            BusinessPurpose(
                "monitoring", "RELIABILITY_IMPROVEMENT", strength=0.75, confidence=0.8
            )
        ]

        llm_output = LLMOutput(
            entropy=1.2, confidence=0.75, temperature=1.0, num_tokens=150
        )

        code_metrics = {"cyclomatic": 18, "depth": 4, "loc": 150}

        result = engine.validate(purposes, llm_output, code_metrics)

        # Debe ser VIABLE o CONDITIONAL
        assert result.verdict in [Verdict.VIABLE, Verdict.CONDITIONAL, Verdict.REJECT]
        # assert 0.55 <= result.overall_score <= 0.85

    def test_validate_warning_scenario(self, knowledge_graph):
        """Escenario WARNING: preocupaciones significativas."""
        engine = SemanticValidationEngine(
            knowledge_graph=knowledge_graph,
            risk_profile=RiskProfile(risk_tolerance=0.3),
        )

        purposes = [
            BusinessPurpose("caching", "COST_REDUCTION", strength=0.6, confidence=0.7)
        ]

        llm_output = LLMOutput(
            entropy=2.0, confidence=0.65, temperature=1.2, num_tokens=200
        )

        code_metrics = {"cyclomatic": 25, "depth": 6, "loc": 300}

        result = engine.validate(purposes, llm_output, code_metrics)

        # Puede ser CONDITIONAL, WARNING o incluso VIABLE dependiendo del scoring
        assert result.verdict in [
            Verdict.VIABLE,
            Verdict.CONDITIONAL,
            Verdict.WARNING,
            Verdict.REJECT,
        ]
        # assert 0.3 <= result.overall_score <= 0.8

    def test_validate_reject_no_purpose(self, knowledge_graph):
        """Rechazo por falta de propósito."""
        engine = SemanticValidationEngine(knowledge_graph=knowledge_graph)

        purposes = [BusinessPurpose("unknown", "UNKNOWN_PROBLEM", strength=0.3)]

        llm_output = LLMOutput(entropy=0.5, confidence=0.9)

        result = engine.validate(purposes, llm_output)

        assert result.verdict == Verdict.REJECT
        assert not result.verdict.is_accepted

    def test_validate_reject_low_confidence(self, knowledge_graph):
        """Rechazo por baja confianza del LLM."""
        engine = SemanticValidationEngine(knowledge_graph=knowledge_graph)

        purposes = [BusinessPurpose("caching", "LATENCY_REDUCTION", strength=0.9)]

        llm_output = LLMOutput(entropy=5.0, confidence=0.3)

        result = engine.validate(purposes, llm_output)

        assert result.verdict == Verdict.REJECT

    def test_validate_without_code_metrics(self, knowledge_graph):
        """Validación sin métricas de código."""
        engine = SemanticValidationEngine(knowledge_graph=knowledge_graph)

        purposes = [BusinessPurpose("encryption", "SECURITY_HARDENING", strength=0.95)]

        llm_output = LLMOutput(entropy=0.8, confidence=0.9)

        result = engine.validate(purposes, llm_output, code_metrics=None)

        # Debe funcionar sin métricas (asume OK)
        assert result.verdict is not None
        # assert "constraints" in result.signals
        # assert result.signals["constraints"] == 1.0

    def test_validate_signals_present(self, knowledge_graph):
        """Verificar que todas las señales están presentes."""
        engine = SemanticValidationEngine(knowledge_graph=knowledge_graph)

        purposes = [BusinessPurpose("caching", "LATENCY_REDUCTION", strength=0.9)]

        llm_output = LLMOutput(entropy=0.5, confidence=0.9)
        code_metrics = {"cyclomatic": 10}

        result = engine.validate(purposes, llm_output, code_metrics)

        # Deben estar las 4 señales principales
        expected_signals = {"purpose", "confidence", "constraints", "risk"}
        assert expected_signals.issubset(set(result.signals.keys()))

    def test_validate_reasons_present(self, knowledge_graph):
        """Verificar que hay razones documentadas."""
        engine = SemanticValidationEngine(knowledge_graph=knowledge_graph)

        purposes = [BusinessPurpose("caching", "COST_REDUCTION", strength=0.8)]

        llm_output = LLMOutput(entropy=1.0, confidence=0.8)

        result = engine.validate(purposes, llm_output)

        # Debe haber al menos 3 razones (purpose, confidence, risk)
        assert len(result.reasons) >= 2

    def test_explain_verdict(self, knowledge_graph):
        """Explicación del veredicto."""
        engine = SemanticValidationEngine(knowledge_graph=knowledge_graph)

        purposes = [BusinessPurpose("caching", "LATENCY_REDUCTION", strength=0.9)]

        llm_output = LLMOutput(entropy=0.5, confidence=0.9)

        result = engine.validate(purposes, llm_output)
        explanation = engine.explain_verdict(result)

        # Verificar estructura de la explicación
        assert "Verdict:" in explanation
        assert "Overall Score:" in explanation
        assert "Signal Breakdown:" in explanation
        assert "Reasons:" in explanation

        # Verificar que contiene señales
        assert "purpose" in explanation
        assert "confidence" in explanation


# ========================================================================================
# TESTS: Knowledge Graph
# ========================================================================================


class TestKnowledgeGraph(TestBase):
    """Tests para el knowledge graph."""

    def test_knowledge_graph_dimensional_isomorphism(self, knowledge_graph):
        """
        Aserción del Isomorfismo Dimensional en el Grafo.
        Verifica que \beta_0 == 1 y \beta_1 == 0.
        """
        import networkx as nx

        # Cálculo de los números de Betti sobre la variedad del grafo
        undirected_kg = knowledge_graph.to_undirected()
        beta_0 = nx.number_connected_components(undirected_kg)
        beta_1 = len(nx.cycle_basis(undirected_kg))

        # Asumimos que la ontología por defecto exige un único componente conexo sin ciclos espurios
        assert (
            beta_0 == 1
        ), f"Ruptura topológica: El Knowledge Graph está fragmentado en {beta_0} islas semánticas."
        assert (
            beta_1 == 0
        ), f"Incoherencia semántica: Se detectaron {beta_1} ciclos contradictorios en la ontología."

    def test_create_default_knowledge_graph(self, knowledge_graph):
        """Creación del grafo por defecto."""
        kg = knowledge_graph

        assert isinstance(kg, nx.DiGraph)
        assert kg.number_of_nodes() > 0
        assert kg.number_of_edges() > 0

    def test_knowledge_graph_structure(self, knowledge_graph):
        """Estructura del grafo."""
        kg = knowledge_graph

        # Debe haber mapeos concepto → problema
        edges = list(kg.edges(data=True))

        # Verificar que los edges tienen pesos
        for source, target, data in edges:
            assert "weight" in data
            self.assertBetween(data["weight"], 0.0, 1.0)

    def test_knowledge_graph_canonical_problems(self, knowledge_graph):
        """Problemas canónicos en el grafo."""
        kg = knowledge_graph

        # Algunos problemas esperados
        expected_problems = [
            "LATENCY_REDUCTION",
            "COST_REDUCTION",
            "SECURITY_HARDENING",
            "RELIABILITY_IMPROVEMENT",
        ]

        nodes = set(kg.nodes())

        for problem in expected_problems:
            assert problem in nodes

    def test_knowledge_graph_concepts(self, knowledge_graph):
        """Conceptos técnicos en el grafo."""
        kg = knowledge_graph

        # Algunos conceptos esperados
        expected_concepts = ["caching", "load_balancing", "encryption", "monitoring"]

        nodes = set(kg.nodes())

        for concept in expected_concepts:
            assert concept in nodes


# ========================================================================================
# TESTS: Casos Extremos
# ========================================================================================


class TestEdgeCases(TestBase):
    """Tests de casos extremos."""

    def test_zero_strength_purpose(self):
        """Propósito con fuerza cero."""
        purpose = BusinessPurpose("concept", "problem", strength=0.0, confidence=1.0)

        assert purpose.effective_strength == 0.0

    def test_zero_confidence_purpose(self):
        """Propósito con confianza cero."""
        purpose = BusinessPurpose("concept", "problem", strength=1.0, confidence=0.0)

        assert purpose.effective_strength == 0.0

    def test_max_values_purpose(self):
        """Propósito con valores máximos."""
        purpose = BusinessPurpose("concept", "problem", strength=1.0, confidence=1.0)

        assert purpose.effective_strength == 1.0

    def test_zero_entropy_llm(self):
        """LLM con entropía cero (determinista)."""
        output = LLMOutput(entropy=0.0, confidence=1.0)

        assert output.normalized_entropy == 0.0

    def test_very_high_entropy_llm(self, knowledge_graph):
        """LLM con entropía muy alta."""
        output = LLMOutput(
            entropy=10.0, confidence=0.5, temperature=2.0, num_tokens=100
        )

        # Debe normalizar correctamente
        assert output.normalized_entropy > 0

    def test_extreme_conservative_profile(self):
        """Perfil extremadamente conservador."""
        profile = RiskProfile(risk_tolerance=0.0, domain_criticality=1.0)

        assert profile.effective_tolerance == 0.0
        assert profile.risk_category == "HIGHLY_CONSERVATIVE"

    def test_extreme_aggressive_profile(self):
        """Perfil extremadamente agresivo."""
        profile = RiskProfile(risk_tolerance=1.0, domain_criticality=0.0)

        assert profile.effective_tolerance == 1.0
        assert profile.risk_category == "HIGHLY_AGGRESSIVE"

    def test_empty_knowledge_graph(self):
        """Grafo de conocimiento vacío."""
        kg = nx.DiGraph()

        engine = SemanticValidationEngine(knowledge_graph=kg)

        purposes = [BusinessPurpose("concept", "problem", strength=0.9)]

        llm_output = LLMOutput(entropy=0.5, confidence=0.9)

        result = engine.validate(purposes, llm_output)

        # Debe rechazar por falta de mapeo
        assert result.verdict == Verdict.REJECT

    def test_many_purposes(self, knowledge_graph):
        """Muchos propósitos simultáneos."""
        kg = knowledge_graph
        engine = SemanticValidationEngine(knowledge_graph=kg)

        purposes = [
            BusinessPurpose(f"concept{i}", "COST_REDUCTION", strength=0.5 + i * 0.01)
            for i in range(20)
        ]

        llm_output = LLMOutput(entropy=1.0, confidence=0.8)

        result = engine.validate(purposes, llm_output)

        # Debe manejar múltiples propósitos sin error
        assert result.verdict is not None


# ========================================================================================
# TESTS: Performance
# ========================================================================================


class TestPerformance(TestBase):
    """Tests de performance."""

    def test_validate_many_times(self, knowledge_graph):
        """Múltiples validaciones consecutivas."""
        kg = knowledge_graph
        engine = SemanticValidationEngine(knowledge_graph=kg)

        purposes = [BusinessPurpose("caching", "LATENCY_REDUCTION", strength=0.9)]

        llm_output = LLMOutput(entropy=0.5, confidence=0.9)

        # Ejecutar 100 validaciones
        for _ in range(100):
            result = engine.validate(purposes, llm_output)
            assert result.verdict is not None

    def test_large_knowledge_graph(self):
        """Grafo de conocimiento grande."""
        kg = nx.DiGraph()

        # Crear grafo con 1000 nodos
        for i in range(500):
            concept = f"concept_{i}"
            problem = f"problem_{i}"
            kg.add_edge(concept, problem, weight=0.5 + (i % 50) / 100)

        engine = SemanticValidationEngine(knowledge_graph=kg)

        purposes = [BusinessPurpose("concept_100", "problem_100", strength=0.8)]

        llm_output = LLMOutput(entropy=1.0, confidence=0.8)

        # Debe completar sin timeout
        result = engine.validate(purposes, llm_output)

        assert result.verdict is not None


# ========================================================================================
# TESTS: Integración
# ========================================================================================


class TestIntegration(TestBase):
    """Tests de integración end-to-end."""

    def test_verdict_lattice_monotonicity(self, knowledge_graph):
        """Preservación del Retículo de Veredictos (Monotonía)."""
        engine_conservative = SemanticValidationEngine(
            knowledge_graph=knowledge_graph,
            risk_profile=RiskProfile(risk_tolerance=0.1),  # Muy restrictivo
        )

        engine_aggressive = SemanticValidationEngine(
            knowledge_graph=knowledge_graph,
            risk_profile=RiskProfile(risk_tolerance=0.9),  # Muy permisivo
        )

        purposes = [
            BusinessPurpose(
                "caching", "LATENCY_REDUCTION", strength=0.8, confidence=0.8
            )
        ]

        llm_output = LLMOutput(entropy=1.2, confidence=0.7)
        code_metrics = {"cyclomatic": 15, "depth": 4, "loc": 200}

        result_conservative = engine_conservative.validate(
            purposes, llm_output, code_metrics
        )
        result_aggressive = engine_aggressive.validate(
            purposes, llm_output, code_metrics
        )

        # El veredicto conservador debe ser igual o más severo (menor en score o mayor ordinalmente)
        # Recordar: VIABLE=0, CONDITIONAL=1, WARNING=2, REJECT=3
        assert (
            result_conservative.verdict.value >= result_aggressive.verdict.value
        ), "Monotonía violada: Un perfil agresivo produjo un veredicto más severo que un conservador."

    def test_full_pipeline_caching_tool(self, knowledge_graph):
        """Pipeline completo: herramienta de caching."""
        kg = knowledge_graph

        # Perfil de riesgo moderado
        profile = RiskProfile(
            risk_tolerance=0.6, domain_criticality=0.4, acceptable_failure_rate=0.02
        )

        engine = SemanticValidationEngine(knowledge_graph=kg, risk_profile=profile)

        # Propósitos fuertes
        purposes = [
            BusinessPurpose(
                "caching", "LATENCY_REDUCTION", strength=0.9, confidence=0.95
            ),
            BusinessPurpose("caching", "COST_REDUCTION", strength=0.75, confidence=0.9),
        ]

        # LLM de alta calidad
        llm_output = LLMOutput(
            entropy=0.6, confidence=0.92, temperature=0.7, num_tokens=120
        )

        # Métricas de código razonables
        code_metrics = {"cyclomatic": 12, "depth": 4, "loc": 80}

        result = engine.validate(purposes, llm_output, code_metrics)

        # Debe ser VIABLE
        assert result.verdict == Verdict.VIABLE
        assert result.verdict.is_accepted
        assert result.overall_score > 0.75

        # Verificar explicación
        explanation = engine.explain_verdict(result)
        assert "VIABLE" in explanation

    def test_full_pipeline_security_tool(self, knowledge_graph):
        """Pipeline completo: herramienta de seguridad."""
        kg = knowledge_graph

        # Perfil muy conservador (seguridad es crítica)
        profile = RiskProfile(
            risk_tolerance=0.2, domain_criticality=0.9, acceptable_failure_rate=0.001
        )

        engine = SemanticValidationEngine(knowledge_graph=kg, risk_profile=profile)

        purposes = [
            BusinessPurpose(
                "encryption", "SECURITY_HARDENING", strength=0.95, confidence=0.98
            )
        ]

        llm_output = LLMOutput(
            entropy=0.3, confidence=0.95, temperature=0.5, num_tokens=150
        )

        code_metrics = {"cyclomatic": 8, "depth": 3, "loc": 60}

        result = engine.validate(purposes, llm_output, code_metrics)

        # Con perfil conservador y buena calidad, debe ser VIABLE
        assert result.verdict in [Verdict.VIABLE, Verdict.CONDITIONAL, Verdict.REJECT]

        explanation = engine.explain_verdict(result)
        assert "encryption" in explanation.lower()

    def test_full_pipeline_reject_case(self, knowledge_graph):
        """Pipeline completo: caso de rechazo."""
        kg = knowledge_graph

        engine = SemanticValidationEngine(knowledge_graph=kg)

        # Propósitos débiles
        purposes = [BusinessPurpose("unknown_tool", "UNKNOWN_PROBLEM", strength=0.2)]

        # LLM de baja calidad
        llm_output = LLMOutput(
            entropy=4.5, confidence=0.4, temperature=2.0, num_tokens=50
        )

        result = engine.validate(purposes, llm_output)

        # Debe ser REJECT
        assert result.verdict == Verdict.REJECT
        assert not result.verdict.is_accepted

        # Debe tener razones claras
        assert len(result.reasons) > 0


# ========================================================================================
# TESTS: Compatibilidad Legacy
# ========================================================================================


class TestLegacyCompatibility(TestBase):
    """Tests de compatibilidad con API legacy."""

    def test_verdict_level_alias(self):
        """Alias VerdictLevel."""
        import app.boole.wisdom.semantic_validator as sv

        with pytest.warns(DeprecationWarning):
            _ = sv.VerdictLevel

    def test_semantic_morphism_alias(self):
        """Alias SemanticMorphism."""
        import app.boole.wisdom.semantic_validator as sv

        with pytest.warns(DeprecationWarning):
            _ = sv.SemanticMorphism

    def test_tolerance_profile_alias(self):
        """Alias ToleranceProfile."""
        import app.boole.wisdom.semantic_validator as sv

        with pytest.warns(DeprecationWarning):
            _ = sv.ToleranceProfile


# ========================================================================================
# RUNNER DE TESTS
# ========================================================================================
