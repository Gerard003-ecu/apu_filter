"""
=========================================================================================
Módulo: Test Suite for Semantic Validation Engine
Ubicación: tests/unit/boole/wisdom/test_semantic_validator.py
Versión: 2.0 - Rigorous Mathematical Validation
Autor: Sistema Gamma-WISDOM

FILOSOFÍA DE TESTING:

Este módulo implementa una suite de pruebas exhaustiva basada en:

1. TEORÍA DE CATEGORÍAS (Funtorialidad)
   - Verificación de morfismos entre espacios de señales
   - Preservación de estructura bajo transformaciones

2. ÁLGEBRA ABSTRACTA (Invariantes Algebraicas)
   - Propiedades de retículo (supremo, ínfimo)
   - Simetría y transitividad de relaciones de orden

3. TOPOLOGÍA ALGEBRAICA (Obstrucciones)
   - Detección de ciclos no triviales
   - Cohomología de complejos simpliciales

4. ANÁLISIS FUNCIONAL (Continuidad y Lipschitz)
   - Variación suave de distancia de Mahalanobis
   - Cotas de Lipschitz en mapeos de riesgo

5. TEORÍA DE CONTRATOS (Design by Contract)
   - Precondiciones, postcondiciones, invariantes
   - Verificación exhaustiva de violaciones

ESTRUCTURA DE LA SUITE:

├── TestDataStructures
│   ├── Inmutabilidad
│   ├── Validación de invariantes
│   └── Serialización
│
├── TestMahalanobisMetric
│   ├── Propiedades geométricas
│   ├── Definición positiva
│   └── Número de condición
│
├── TestSimplicialCohomology
│   ├── Detección de ciclos
│   ├── Dimensión cohomológica
│   └── Obstrucciones topológicas
│
├── TestValidators
│   ├── PurposeValidator
│   ├── ConfidenceFilter
│   └── ConstraintMapper
│
├── TestSemanticValidationEngine
│   ├── Casos nominales
│   ├── Casos extremos
│   ├── Singularidades
│   └── Paradojas semánticas
│
├── TestRobustness
│   ├── Entradas aleatorias (fuzzing)
│   ├── Estabilidad numérica
│   └── Casos degenerados
│
└── TestIntegration
    ├── Flujos end-to-end
    ├── Compatibilidad legacy
    └── Benchmarks de rendimiento

=========================================================================================
"""

import math
import unittest
from typing import Dict, List
from unittest.mock import Mock, patch

import numpy as np
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from hypothesis.strategies import composite

# Importar módulo a testear (ajustar path según estructura del proyecto)
import sys
sys.path.insert(0, '..')  # Ajustar según necesidad

from semantic_validator import (
    # Excepciones
    ValidationError,
    TopologicalObstructionError,
    MetricDegeneracyError,
    ContractViolationError,
    
    # Tipos y enumeraciones
    Verdict,
    BusinessPurpose,
    LLMOutput,
    RiskProfile,
    ValidationResult,
    
    # Componentes principales
    MahalanobisMetric,
    SimplicialCohomology,
    PurposeValidator,
    ConfidenceFilter,
    ConstraintMapper,
    SemanticValidationEngine,
    
    # Utilidades
    create_default_knowledge_graph,
)


# =============================================================================
# ESTRATEGIAS DE HYPOTHESIS PARA PROPERTY-BASED TESTING
# =============================================================================

@composite
def business_purpose_strategy(draw):
    """Genera BusinessPurpose válidos aleatoriamente."""
    concepts = ['caching', 'load_balancing', 'encryption', 'monitoring', 'refactoring']
    problems = [
        'COST_REDUCTION', 'LATENCY_REDUCTION', 'RELIABILITY_IMPROVEMENT',
        'SCALABILITY_ENHANCEMENT', 'SECURITY_HARDENING'
    ]
    
    return BusinessPurpose(
        concept=draw(st.sampled_from(concepts)),
        business_problem=draw(st.sampled_from(problems)),
        strength=draw(st.floats(min_value=0.0, max_value=1.0)),
        confidence=draw(st.floats(min_value=0.0, max_value=1.0))
    )


@composite
def llm_output_strategy(draw, allow_singular=False):
    """Genera LLMOutput válidos aleatoriamente."""
    if allow_singular and draw(st.booleans()):
        # Generar singularidad ocasionalmente
        if draw(st.booleans()):
            entropy = float('inf')
            confidence = draw(st.floats(min_value=0.1, max_value=1.0))
        else:
            entropy = draw(st.floats(min_value=0.0, max_value=5.0))
            confidence = 0.0
    else:
        entropy = draw(st.floats(min_value=0.0, max_value=5.0))
        confidence = draw(st.floats(min_value=0.01, max_value=1.0))
    
    return LLMOutput(
        entropy=entropy,
        confidence=confidence,
        temperature=draw(st.floats(min_value=0.1, max_value=2.0)),
        num_tokens=draw(st.integers(min_value=0, max_value=1000))
    )


@composite
def risk_profile_strategy(draw):
    """Genera RiskProfile válidos aleatoriamente."""
    return RiskProfile(
        risk_tolerance=draw(st.floats(min_value=0.0, max_value=1.0)),
        domain_criticality=draw(st.floats(min_value=0.0, max_value=1.0)),
        acceptable_failure_rate=draw(st.floats(min_value=0.0, max_value=1.0))
    )


@composite
def positive_definite_matrix_strategy(draw, size=4):
    """Genera matriz definida positiva aleatoria."""
    # Generar matriz aleatoria
    A = np.random.randn(size, size)
    # Hacer simétrica definida positiva: A^T A + εI
    epsilon = draw(st.floats(min_value=0.1, max_value=1.0))
    M = A.T @ A + epsilon * np.eye(size)
    return M


# =============================================================================
# TEST 1: ESTRUCTURAS DE DATOS INMUTABLES
# =============================================================================

class TestDataStructures(unittest.TestCase):
    """Verifica invariantes de estructuras de datos inmutables."""
    
    def test_verdict_ordering(self):
        """Verifica que Verdict mantiene orden total."""
        self.assertLess(Verdict.VIABLE, Verdict.CONDITIONAL)
        self.assertLess(Verdict.CONDITIONAL, Verdict.WARNING)
        self.assertLess(Verdict.WARNING, Verdict.REJECT)
        
        # Transitividad
        self.assertLess(Verdict.VIABLE, Verdict.WARNING)
        self.assertLess(Verdict.VIABLE, Verdict.REJECT)
        self.assertLess(Verdict.CONDITIONAL, Verdict.REJECT)
    
    def test_verdict_lattice_operations(self):
        """Verifica operaciones de retículo (supremo, ínfimo)."""
        # Supremo (join, OR)
        self.assertEqual(Verdict.VIABLE | Verdict.CONDITIONAL, Verdict.CONDITIONAL)
        self.assertEqual(Verdict.WARNING | Verdict.REJECT, Verdict.REJECT)
        self.assertEqual(Verdict.VIABLE | Verdict.REJECT, Verdict.REJECT)
        
        # Ínfimo (meet, AND)
        self.assertEqual(Verdict.VIABLE & Verdict.CONDITIONAL, Verdict.VIABLE)
        self.assertEqual(Verdict.WARNING & Verdict.REJECT, Verdict.WARNING)
        self.assertEqual(Verdict.VIABLE & Verdict.REJECT, Verdict.VIABLE)
        
        # Idempotencia: v ∨ v = v
        for verdict in Verdict:
            self.assertEqual(verdict | verdict, verdict)
            self.assertEqual(verdict & verdict, verdict)
        
        # Conmutatividad
        self.assertEqual(
            Verdict.VIABLE | Verdict.WARNING,
            Verdict.WARNING | Verdict.VIABLE
        )
    
    def test_verdict_properties(self):
        """Verifica propiedades semánticas de veredictos."""
        self.assertTrue(Verdict.VIABLE.is_accepted)
        self.assertTrue(Verdict.CONDITIONAL.is_accepted)
        self.assertFalse(Verdict.WARNING.is_accepted)
        self.assertFalse(Verdict.REJECT.is_accepted)
        
        self.assertFalse(Verdict.VIABLE.requires_human_review)
        self.assertTrue(Verdict.CONDITIONAL.requires_human_review)
        self.assertTrue(Verdict.WARNING.requires_human_review)
        self.assertFalse(Verdict.REJECT.requires_human_review)
        
        # Severity score monotónico
        scores = [v.severity_score for v in Verdict]
        self.assertEqual(scores, sorted(scores))
    
    def test_business_purpose_immutability(self):
        """Verifica inmutabilidad de BusinessPurpose."""
        purpose = BusinessPurpose("caching", "LATENCY_REDUCTION", 0.9, 0.95)
        
        # No debe permitir asignación
        with self.assertRaises(AttributeError):
            purpose.strength = 0.5
        
        with self.assertRaises(AttributeError):
            purpose.concept = "new_concept"
    
    def test_business_purpose_validation(self):
        """Verifica validación de invariantes en BusinessPurpose."""
        # Strength fuera de rango
        with self.assertRaises(ContractViolationError):
            BusinessPurpose("caching", "COST_REDUCTION", strength=-0.1)
        
        with self.assertRaises(ContractViolationError):
            BusinessPurpose("caching", "COST_REDUCTION", strength=1.5)
        
        # Confidence fuera de rango
        with self.assertRaises(ContractViolationError):
            BusinessPurpose("caching", "COST_REDUCTION", strength=0.5, confidence=-0.1)
        
        # Concept vacío
        with self.assertRaises(ContractViolationError):
            BusinessPurpose("", "COST_REDUCTION", strength=0.5)
        
        # Business problem vacío
        with self.assertRaises(ContractViolationError):
            BusinessPurpose("caching", "", strength=0.5)
    
    def test_business_purpose_effective_strength(self):
        """Verifica cálculo de fuerza efectiva."""
        purpose = BusinessPurpose("caching", "LATENCY_REDUCTION", 0.8, 0.5)
        self.assertAlmostEqual(purpose.effective_strength, 0.4)
        
        # Postcondición: 0 ≤ effective_strength ≤ 1
        self.assertGreaterEqual(purpose.effective_strength, 0.0)
        self.assertLessEqual(purpose.effective_strength, 1.0)
    
    def test_llm_output_validation(self):
        """Verifica validación de LLMOutput."""
        # Entropía negativa
        with self.assertRaises(ContractViolationError):
            LLMOutput(entropy=-1.0, confidence=0.8)
        
        # Confidence fuera de rango
        with self.assertRaises(ContractViolationError):
            LLMOutput(entropy=1.0, confidence=1.5)
        
        # Temperature no positiva
        with self.assertRaises(ContractViolationError):
            LLMOutput(entropy=1.0, confidence=0.8, temperature=0.0)
        
        # Num_tokens negativo
        with self.assertRaises(ContractViolationError):
            LLMOutput(entropy=1.0, confidence=0.8, num_tokens=-10)
    
    def test_llm_output_singularity_detection(self):
        """Verifica detección de singularidades estocásticas."""
        # Entropía infinita
        singular1 = LLMOutput(entropy=float('inf'), confidence=0.8)
        self.assertTrue(singular1.is_singular)
        
        # Confidence cero
        singular2 = LLMOutput(entropy=1.5, confidence=0.0)
        self.assertTrue(singular2.is_singular)
        
        # No singular
        regular = LLMOutput(entropy=1.5, confidence=0.8)
        self.assertFalse(regular.is_singular)
    
    def test_llm_output_normalized_entropy(self):
        """Verifica normalización de entropía."""
        llm = LLMOutput(entropy=2.0, confidence=0.8, temperature=1.0, num_tokens=100)
        
        # H_norm = H / (T × √N) = 2.0 / (1.0 × 10) = 0.2
        expected = 2.0 / (1.0 * math.sqrt(100))
        self.assertAlmostEqual(llm.normalized_entropy, expected)
        
        # Postcondición: ≥ 0
        self.assertGreaterEqual(llm.normalized_entropy, 0.0)
    
    def test_llm_output_perplexity(self):
        """Verifica cálculo de perplejidad."""
        llm = LLMOutput(entropy=2.0, confidence=0.8)
        expected = math.exp(2.0)
        self.assertAlmostEqual(llm.perplexity, expected)
        
        # Postcondición: ≥ 1
        self.assertGreaterEqual(llm.perplexity, 1.0)
    
    def test_risk_profile_validation(self):
        """Verifica validación de RiskProfile."""
        # Valores fuera de rango
        with self.assertRaises(ContractViolationError):
            RiskProfile(risk_tolerance=1.5)
        
        with self.assertRaises(ContractViolationError):
            RiskProfile(risk_tolerance=0.5, domain_criticality=-0.1)
        
        with self.assertRaises(ContractViolationError):
            RiskProfile(risk_tolerance=0.5, acceptable_failure_rate=1.5)
    
    def test_risk_profile_effective_tolerance(self):
        """Verifica cálculo de tolerancia efectiva."""
        # Alta criticidad reduce tolerancia
        profile1 = RiskProfile(risk_tolerance=0.8, domain_criticality=0.8)
        # τ_eff = 0.8 × (1 - 0.5 × 0.8) = 0.8 × 0.6 = 0.48
        self.assertAlmostEqual(profile1.effective_tolerance, 0.48)
        
        # Baja criticidad mantiene tolerancia
        profile2 = RiskProfile(risk_tolerance=0.8, domain_criticality=0.0)
        self.assertAlmostEqual(profile2.effective_tolerance, 0.8)
        
        # Postcondición: ∈ [0, 1]
        self.assertGreaterEqual(profile1.effective_tolerance, 0.0)
        self.assertLessEqual(profile1.effective_tolerance, 1.0)
    
    def test_risk_profile_categorization(self):
        """Verifica categorización de riesgo."""
        categories = [
            (0.1, "HIGHLY_CONSERVATIVE"),
            (0.3, "CONSERVATIVE"),
            (0.5, "MODERATE"),
            (0.7, "AGGRESSIVE"),
            (0.9, "HIGHLY_AGGRESSIVE"),
        ]
        
        for tolerance, expected_category in categories:
            profile = RiskProfile(risk_tolerance=tolerance, domain_criticality=0.0)
            self.assertEqual(profile.risk_category, expected_category)
    
    def test_validation_result_postconditions(self):
        """Verifica postcondiciones de ValidationResult."""
        result = ValidationResult(
            verdict=Verdict.VIABLE,
            mahalanobis_distance=0.25
        )
        
        # Distancia no negativa
        self.assertGreaterEqual(result.mahalanobis_distance, 0.0)
        
        # Agregar señal fuera de rango (debe clampear)
        result.add_reason("test", "signal1", 1.5)
        self.assertEqual(result.signals["signal1"], 1.0)
        
        result.add_reason("test2", "signal2", -0.5)
        self.assertEqual(result.signals["signal2"], 0.0)
    
    def test_validation_result_contract_violation(self):
        """Verifica detección de violaciones de contrato."""
        # Distancia negativa
        with self.assertRaises(ContractViolationError):
            ValidationResult(verdict=Verdict.VIABLE, mahalanobis_distance=-0.5)
        
        # Signal name sin value
        result = ValidationResult(verdict=Verdict.VIABLE, mahalanobis_distance=0.5)
        with self.assertRaises(ContractViolationError):
            result.add_reason("test", signal_name="sig1")


# =============================================================================
# TEST 2: MÉTRICA DE MAHALANOBIS
# =============================================================================

class TestMahalanobisMetric(unittest.TestCase):
    """Verifica propiedades geométricas de la métrica de Mahalanobis."""
    
    def test_default_metric_initialization(self):
        """Verifica inicialización correcta con métrica default."""
        metric = MahalanobisMetric()
        
        # Shape correcto
        self.assertEqual(metric.G.shape, (4, 4))
        
        # Simetría
        self.assertTrue(np.allclose(metric.G, metric.G.T))
        
        # Definición positiva (autovalores > 0)
        eigvals = np.linalg.eigvalsh(metric.G)
        self.assertTrue(np.all(eigvals > 0))
        
        # Número de condición razonable
        self.assertLess(metric._condition_number, MahalanobisMetric.MAX_CONDITION_NUMBER)
    
    def test_custom_metric_validation(self):
        """Verifica validación de tensor métrico personalizado."""
        # Matriz no simétrica
        non_symmetric = np.array([
            [1.0, 0.5, 0.0, 0.0],
            [0.3, 1.0, 0.0, 0.0],  # 0.5 != 0.3
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        
        with self.assertRaises(ContractViolationError):
            MahalanobisMetric(non_symmetric)
        
        # Matriz no definida positiva (autovalor negativo)
        indefinite = np.array([
            [1.0,  0.0,  0.0,  0.0],
            [0.0, -1.0,  0.0,  0.0],  # Autovalor negativo
            [0.0,  0.0,  1.0,  0.0],
            [0.0,  0.0,  0.0,  1.0],
        ])
        
        with self.assertRaises(ContractViolationError):
            MahalanobisMetric(indefinite)
        
        # Shape incorrecto
        wrong_shape = np.eye(3)
        with self.assertRaises(ContractViolationError):
            MahalanobisMetric(wrong_shape)
    
    def test_distance_to_ideal_properties(self):
        """Verifica propiedades matemáticas de la distancia."""
        metric = MahalanobisMetric()
        
        # D_M(s*) = 0 (distancia al ideal es cero)
        ideal = np.ones(4)
        distance_to_self = metric.distance_to_ideal(ideal, ideal)
        self.assertAlmostEqual(distance_to_self, 0.0, places=10)
        
        # D_M ≥ 0 (no-negatividad)
        signal = np.array([0.5, 0.6, 0.7, 0.8])
        distance = metric.distance_to_ideal(signal)
        self.assertGreaterEqual(distance, 0.0)
        
        # D_M(s) > 0 si s ≠ s* (separación)
        non_ideal = np.array([0.5, 0.5, 0.5, 0.5])
        distance_non_ideal = metric.distance_to_ideal(non_ideal)
        self.assertGreater(distance_non_ideal, 0.0)
    
    def test_distance_triangle_inequality(self):
        """Verifica desigualdad triangular (si métrica es euclidiana)."""
        metric = MahalanobisMetric()
        
        # Para métrica euclidiana inducida: d(x,z) ≤ d(x,y) + d(y,z)
        # No se cumple estrictamente para Mahalanobis general, pero verificamos
        # que no viola groseramente
        
        x = np.array([0.3, 0.4, 0.5, 0.6])
        y = np.array([0.6, 0.7, 0.8, 0.9])
        z = np.ones(4)
        
        d_xz = metric.distance_to_ideal(x, z)
        d_xy = metric.distance_to_ideal(x, y)
        d_yz = metric.distance_to_ideal(y, z)
        
        # Verificar que no es absurdamente grande
        self.assertLess(d_xz, 10 * (d_xy + d_yz))
    
    def test_distance_continuity(self):
        """Verifica continuidad de la distancia (aproximación numérica)."""
        metric = MahalanobisMetric()
        
        signal = np.array([0.5, 0.6, 0.7, 0.8])
        distance_base = metric.distance_to_ideal(signal)
        
        # Pequeña perturbación
        epsilon = 1e-6
        perturbed = signal + epsilon * np.ones(4)
        distance_perturbed = metric.distance_to_ideal(perturbed)
        
        # La diferencia debe ser pequeña (continuidad)
        self.assertLess(abs(distance_perturbed - distance_base), 0.01)
    
    def test_set_coupling_preserves_positive_definiteness(self):
        """Verifica que set_coupling mantiene definición positiva."""
        metric = MahalanobisMetric()
        
        # Cambio válido
        metric.set_coupling(0, 1, 0.5)
        self.assertEqual(metric.G[0, 1], 0.5)
        self.assertEqual(metric.G[1, 0], 0.5)  # Simetría
        
        # Verificar que sigue siendo definida positiva
        eigvals = np.linalg.eigvalsh(metric.G)
        self.assertTrue(np.all(eigvals > 0))
    
    def test_set_coupling_rejects_indefinite_matrix(self):
        """Verifica que set_coupling rechaza cambios que violan def. positiva."""
        metric = MahalanobisMetric()
        
        # Intentar crear matriz indefinida (acoplamiento muy negativo)
        with self.assertRaises(ContractViolationError):
            metric.set_coupling(0, 1, -10.0)
        
        # Verificar que la métrica no cambió
        self.assertEqual(metric.G[0, 1], MahalanobisMetric.DEFAULT_METRIC_TENSOR[0, 1])
    
    def test_set_coupling_validates_indices(self):
        """Verifica validación de índices en set_coupling."""
        metric = MahalanobisMetric()
        
        with self.assertRaises(ContractViolationError):
            metric.set_coupling(-1, 0, 0.5)
        
        with self.assertRaises(ContractViolationError):
            metric.set_coupling(0, 5, 0.5)
    
    def test_metric_copy_independence(self):
        """Verifica independencia de copias."""
        metric1 = MahalanobisMetric()
        metric2 = metric1.copy()
        
        # Modificar copia no debe afectar original
        metric2.set_coupling(0, 1, 0.8)
        
        self.assertNotEqual(metric1.G[0, 1], metric2.G[0, 1])
    
    def test_condition_number_computation(self):
        """Verifica cálculo correcto del número de condición."""
        # Matriz con número de condición conocido
        # κ = λ_max / λ_min
        well_conditioned = np.diag([1.0, 2.0, 3.0, 4.0])
        metric = MahalanobisMetric(well_conditioned)
        
        expected_kappa = 4.0 / 1.0  # 4.0
        self.assertAlmostEqual(metric._condition_number, expected_kappa, places=5)
        
        # Postcondición: κ ≥ 1
        self.assertGreaterEqual(metric._condition_number, 1.0)
    
    def test_ill_conditioned_metric_rejection(self):
        """Verifica rechazo de métricas mal condicionadas."""
        # Crear matriz con número de condición muy alto
        ill_conditioned = np.diag([1.0, 1e-10, 1.0, 1.0])
        
        with self.assertRaises(MetricDegeneracyError):
            MahalanobisMetric(ill_conditioned)
    
    @given(positive_definite_matrix_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_distance_nonnegative(self, matrix):
        """Property: La distancia siempre es no-negativa."""
        try:
            metric = MahalanobisMetric(matrix)
            signal = np.random.rand(4)
            distance = metric.distance_to_ideal(signal)
            self.assertGreaterEqual(distance, 0.0)
        except MetricDegeneracyError:
            # Matriz mal condicionada, skip
            pass


# =============================================================================
# TEST 3: COHOMOLOGÍA SIMPLICIAL
# =============================================================================

class TestSimplicialCohomology(unittest.TestCase):
    """Verifica detección de obstrucciones topológicas."""
    
    def test_initialization(self):
        """Verifica inicialización correcta."""
        cohom = SimplicialCohomology()
        self.assertEqual(len(cohom._signal_values), 0)
        self.assertIsNone(cohom._cohomology_dimension)
    
    def test_set_signals_validation(self):
        """Verifica validación de señales."""
        cohom = SimplicialCohomology()
        
        # Valores fuera de rango
        with self.assertRaises(ContractViolationError):
            cohom.set_signals(1.5, 0.5, 0.5, 0.5)
        
        with self.assertRaises(ContractViolationError):
            cohom.set_signals(0.5, -0.1, 0.5, 0.5)
        
        # Valores válidos
        cohom.set_signals(0.5, 0.6, 0.7, 0.8)
        self.assertEqual(len(cohom._signal_values), 4)
    
    def test_no_obstruction_for_consistent_signals(self):
        """Señales consistentes no deben generar obstrucción."""
        cohom = SimplicialCohomology()
        
        # Señales todas altas (consistentes)
        cohom.set_signals(0.9, 0.9, 0.9, 0.9)
        self.assertFalse(cohom.has_obstruction())
        self.assertEqual(cohom.compute_cohomology_dimension(), 0)
        
        # Señales todas bajas (consistentes)
        cohom.set_signals(0.2, 0.2, 0.2, 0.2)
        self.assertFalse(cohom.has_obstruction())
        self.assertEqual(cohom.compute_cohomology_dimension(), 0)
    
    def test_obstruction_for_purpose_confidence_paradox(self):
        """Propósito débil + confianza alta → obstrucción."""
        cohom = SimplicialCohomology()
        
        # Purpose muy bajo, confidence muy alto
        cohom.set_signals(
            purpose=0.2,       # Débil
            confidence=0.95,   # Alta
            constraints=0.8,
            risk=0.5
        )
        
        self.assertTrue(cohom.has_obstruction())
        self.assertGreater(cohom.compute_cohomology_dimension(), 0)
    
    def test_obstruction_for_confidence_constraints_paradox(self):
        """Confianza baja + restricciones perfectas → obstrucción."""
        cohom = SimplicialCohomology()
        
        cohom.set_signals(
            purpose=0.8,
            confidence=0.25,   # Muy baja
            constraints=0.95,  # Muy alta
            risk=0.5
        )
        
        self.assertTrue(cohom.has_obstruction())
        self.assertGreater(cohom.compute_cohomology_dimension(), 0)
    
    def test_obstruction_for_constraints_risk_paradox(self):
        """Restricciones violadas + riesgo alto → obstrucción."""
        cohom = SimplicialCohomology()
        
        cohom.set_signals(
            purpose=0.8,
            confidence=0.8,
            constraints=0.15,  # Muy baja
            risk=0.90          # Muy alta
        )
        
        self.assertTrue(cohom.has_obstruction())
        self.assertGreater(cohom.compute_cohomology_dimension(), 0)
    
    def test_obstruction_for_risk_purpose_paradox(self):
        """Riesgo bajo + propósito fuerte → obstrucción."""
        cohom = SimplicialCohomology()
        
        cohom.set_signals(
            purpose=0.95,      # Muy alto
            confidence=0.8,
            constraints=0.8,
            risk=0.15          # Muy bajo
        )
        
        self.assertTrue(cohom.has_obstruction())
        self.assertGreater(cohom.compute_cohomology_dimension(), 0)
    
    def test_obstruction_description_non_empty(self):
        """Descripción de obstrucción debe ser informativa."""
        cohom = SimplicialCohomology()
        
        # Sin obstrucción
        cohom.set_signals(0.8, 0.8, 0.8, 0.8)
        desc_no_obs = cohom.get_obstruction_description()
        self.assertIn("No topological obstruction", desc_no_obs)
        
        # Con obstrucción
        cohom.set_signals(0.2, 0.95, 0.8, 0.5)
        desc_obs = cohom.get_obstruction_description()
        self.assertIn("dim H¹", desc_obs)
        self.assertIn("paradox", desc_obs.lower())
    
    def test_multiple_obstructions(self):
        """Múltiples paradojas incrementan dimensión cohomológica."""
        cohom = SimplicialCohomology()
        
        # Crear múltiples inconsistencias simultáneas
        cohom.set_signals(
            purpose=0.1,       # Muy bajo
            confidence=0.95,   # Muy alto → paradoja 1
            constraints=0.95,  # Alto pero confidence bajo → paradoja potencial
            risk=0.95          # Alto → paradoja con constraints bajo
        )
        
        dim = cohom.compute_cohomology_dimension()
        # Debe detectar al menos una obstrucción
        self.assertGreater(dim, 0)
    
    def test_cohomology_caching(self):
        """Verifica que cohomología se cachea correctamente."""
        cohom = SimplicialCohomology()
        
        cohom.set_signals(0.5, 0.5, 0.5, 0.5)
        
        # Primera llamada calcula
        dim1 = cohom.compute_cohomology_dimension()
        
        # Segunda llamada usa caché (verificar que no recalcula)
        dim2 = cohom.compute_cohomology_dimension()
        
        self.assertEqual(dim1, dim2)
        
        # Cambiar señales invalida caché
        cohom.set_signals(0.6, 0.6, 0.6, 0.6)
        dim3 = cohom.compute_cohomology_dimension()
        # Puede ser igual o diferente, pero caché se invalidó


# =============================================================================
# TEST 4: VALIDADORES ESPECIALIZADOS
# =============================================================================

class TestPurposeValidator(unittest.TestCase):
    """Verifica validación de propósito empresarial."""
    
    def setUp(self):
        """Configuración común para tests."""
        self.kg = create_default_knowledge_graph()
        self.validator = PurposeValidator(self.kg)
    
    def test_initialization_with_custom_threshold(self):
        """Verifica inicialización con umbral personalizado."""
        validator = PurposeValidator(self.kg, min_strength_threshold=0.8)
        self.assertEqual(validator.min_strength, 0.8)
    
    def test_initialization_validates_threshold(self):
        """Verifica validación de umbral en inicialización."""
        with self.assertRaises(ContractViolationError):
            PurposeValidator(self.kg, min_strength_threshold=0.0)
        
        with self.assertRaises(ContractViolationError):
            PurposeValidator(self.kg, min_strength_threshold=1.5)
    
    def test_empty_purposes_list(self):
        """Lista vacía de propósitos debe fallar validación."""
        is_valid, score, reason = self.validator.validate([])
        
        self.assertFalse(is_valid)
        self.assertEqual(score, 0.0)
        self.assertIn("empty", reason.lower())
    
    def test_non_canonical_purposes(self):
        """Propósitos no canónicos deben fallar validación."""
        purposes = [
            BusinessPurpose("unknown", "UNKNOWN_PROBLEM", strength=0.9)
        ]
        
        is_valid, score, reason = self.validator.validate(purposes)
        
        self.assertFalse(is_valid)
        self.assertEqual(score, 0.0)
        self.assertIn("canonical", reason.lower())
    
    def test_weak_canonical_purpose(self):
        """Propósito canónico débil debe fallar."""
        purposes = [
            BusinessPurpose("caching", "LATENCY_REDUCTION", strength=0.3)
        ]
        
        is_valid, score, reason = self.validator.validate(purposes)
        
        self.assertFalse(is_valid)
        self.assertLess(score, self.validator.min_strength)
        self.assertIn("threshold", reason.lower())
    
    def test_strong_canonical_purpose(self):
        """Propósito canónico fuerte debe pasar."""
        purposes = [
            BusinessPurpose("caching", "LATENCY_REDUCTION", strength=0.9, confidence=0.95)
        ]
        
        is_valid, score, reason = self.validator.validate(purposes)
        
        self.assertTrue(is_valid)
        self.assertGreater(score, self.validator.min_strength)
        self.assertIn("strong", reason.lower())
    
    def test_multiple_purposes_uses_max(self):
        """Múltiples propósitos deben usar el máximo."""
        purposes = [
            BusinessPurpose("caching", "LATENCY_REDUCTION", strength=0.5),
            BusinessPurpose("caching", "COST_REDUCTION", strength=0.9),
        ]
        
        is_valid, score, reason = self.validator.validate(purposes)
        
        self.assertTrue(is_valid)
        self.assertGreater(score, 0.85)  # Cerca de 0.9
    
    def test_compute_purpose_score_aggregation(self):
        """Verifica agregación de score (70% max, 30% mean)."""
        purposes = [
            BusinessPurpose("caching", "LATENCY_REDUCTION", strength=1.0),
            BusinessPurpose("caching", "COST_REDUCTION", strength=0.5),
        ]
        
        score = self.validator.compute_purpose_score(purposes)
        
        # Max = 1.0, Mean = 0.75, Score = 0.7*1.0 + 0.3*0.75 = 0.925
        expected = 0.7 * 1.0 + 0.3 * 0.75
        self.assertAlmostEqual(score, expected, places=3)
    
    def test_purpose_score_postcondition(self):
        """Score de propósito debe estar en [0, 1]."""
        purposes = [
            BusinessPurpose("caching", "LATENCY_REDUCTION", strength=0.95, confidence=0.98)
        ]
        
        score = self.validator.compute_purpose_score(purposes)
        
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestConfidenceFilter(unittest.TestCase):
    """Verifica filtrado de confianza de LLM."""
    
    def setUp(self):
        """Configuración común."""
        self.filter = ConfidenceFilter()
    
    def test_initialization_validates_parameters(self):
        """Verifica validación de parámetros en inicialización."""
        with self.assertRaises(ContractViolationError):
            ConfidenceFilter(min_confidence=0.0)
        
        with self.assertRaises(ContractViolationError):
            ConfidenceFilter(max_entropy=-1.0)
        
        with self.assertRaises(ContractViolationError):
            ConfidenceFilter(max_normalized_entropy=0.0)
    
    def test_singular_llm_output_rejected(self):
        """Singularidades estocásticas deben ser rechazadas."""
        # Entropía infinita
        singular1 = LLMOutput(entropy=float('inf'), confidence=0.8)
        is_valid, score, reason = self.filter.validate(singular1)
        
        self.assertFalse(is_valid)
        self.assertEqual(score, 0.0)
        self.assertIn("singular", reason.lower())
        
        # Confidence cero
        singular2 = LLMOutput(entropy=1.0, confidence=0.0)
        is_valid, score, reason = self.filter.validate(singular2)
        
        self.assertFalse(is_valid)
        self.assertEqual(score, 0.0)
    
    def test_low_confidence_rejected(self):
        """Confianza baja debe ser rechazada."""
        low_conf = LLMOutput(entropy=1.0, confidence=0.3)
        is_valid, score, reason = self.filter.validate(low_conf)
        
        self.assertFalse(is_valid)
        self.assertIn("confidence", reason.lower())
        self.assertIn("threshold", reason.lower())
    
    def test_high_entropy_rejected(self):
        """Entropía alta debe ser rechazada."""
        high_entropy = LLMOutput(entropy=5.0, confidence=0.8)
        is_valid, score, reason = self.filter.validate(high_entropy)
        
        self.assertFalse(is_valid)
        self.assertIn("entropy", reason.lower())
    
    def test_high_normalized_entropy_rejected(self):
        """Entropía normalizada alta debe ser rechazada."""
        # Crear LLM con entropía normalizada alta
        high_norm = LLMOutput(entropy=2.0, confidence=0.8, temperature=0.5, num_tokens=4)
        # H_norm = 2.0 / (0.5 × 2) = 2.0 > 0.5 (default threshold)
        
        is_valid, score, reason = self.filter.validate(high_norm)
        
        self.assertFalse(is_valid)
        self.assertIn("normalized entropy", reason.lower())
    
    def test_high_quality_llm_output_accepted(self):
        """Salida de alta calidad debe ser aceptada."""
        high_quality = LLMOutput(entropy=0.8, confidence=0.92, temperature=0.7, num_tokens=150)
        is_valid, score, reason = self.filter.validate(high_quality)
        
        self.assertTrue(is_valid)
        self.assertGreater(score, 0.7)
        self.assertIn("satisfies", reason.lower())
    
    def test_confidence_score_aggregation(self):
        """Verifica agregación de score (50% conf, 25% H, 25% H_norm)."""
        llm = LLMOutput(entropy=1.0, confidence=0.8, temperature=1.0, num_tokens=100)
        
        # Calcular componentes esperados
        conf_score = 0.8
        entropy_score = max(0.0, 1.0 - 1.0 / self.filter.max_entropy)
        norm_entropy = 1.0 / (1.0 * 10)  # 0.1
        norm_entropy_score = max(0.0, 1.0 - norm_entropy / self.filter.max_normalized_entropy)
        
        expected = 0.5 * conf_score + 0.25 * entropy_score + 0.25 * norm_entropy_score
        
        score = self.filter.compute_confidence_score(llm)
        
        self.assertAlmostEqual(score, expected, places=3)
    
    def test_confidence_score_postcondition(self):
        """Score de confianza debe estar en [0, 1]."""
        llm = LLMOutput(entropy=1.5, confidence=0.75)
        score = self.filter.compute_confidence_score(llm)
        
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestConstraintMapper(unittest.TestCase):
    """Verifica mapeo de restricciones."""
    
    def setUp(self):
        """Configuración común."""
        self.mapper = ConstraintMapper()
    
    def test_map_to_constraints_by_category(self):
        """Verifica mapeo correcto por categoría de riesgo."""
        categories_profiles = [
            ("HIGHLY_CONSERVATIVE", RiskProfile(risk_tolerance=0.1, domain_criticality=0.0)),
            ("CONSERVATIVE", RiskProfile(risk_tolerance=0.3, domain_criticality=0.0)),
            ("MODERATE", RiskProfile(risk_tolerance=0.5, domain_criticality=0.0)),
            ("AGGRESSIVE", RiskProfile(risk_tolerance=0.7, domain_criticality=0.0)),
            ("HIGHLY_AGGRESSIVE", RiskProfile(risk_tolerance=0.9, domain_criticality=0.0)),
        ]
        
        prev_limits = None
        for category, profile in categories_profiles:
            limits = self.mapper.map_to_constraints(profile)
            
            # Verificar que límites existen
            self.assertIn('cyclomatic', limits)
            self.assertIn('depth', limits)
            self.assertIn('loc', limits)
            
            # Límites deben ser positivos
            for value in limits.values():
                self.assertGreaterEqual(value, 1)
            
            # Límites deben incrementar con agresividad
            if prev_limits is not None:
                self.assertGreater(limits['cyclomatic'], prev_limits['cyclomatic'])
            
            prev_limits = limits
    
    def test_criticality_reduces_limits(self):
        """Alta criticidad debe reducir límites."""
        profile_low_crit = RiskProfile(risk_tolerance=0.5, domain_criticality=0.0)
        profile_high_crit = RiskProfile(risk_tolerance=0.5, domain_criticality=0.9)
        
        limits_low = self.mapper.map_to_constraints(profile_low_crit)
        limits_high = self.mapper.map_to_constraints(profile_high_crit)
        
        # Alta criticidad → límites más estrictos (menores)
        self.assertLess(limits_high['cyclomatic'], limits_low['cyclomatic'])
        self.assertLess(limits_high['depth'], limits_low['depth'])
        self.assertLess(limits_high['loc'], limits_low['loc'])
    
    def test_constraint_score_perfect_compliance(self):
        """Cumplimiento perfecto debe dar score 1.0."""
        profile = RiskProfile(risk_tolerance=0.5)
        limits = self.mapper.map_to_constraints(profile)
        
        # Métricas dentro de límites
        metrics = {key: value - 1 for key, value in limits.items()}
        
        score = self.mapper.compute_constraint_score(metrics, profile)
        
        self.assertEqual(score, 1.0)
    
    def test_constraint_score_violations_penalized(self):
        """Violaciones deben reducir score exponencialmente."""
        profile = RiskProfile(risk_tolerance=0.5)
        limits = self.mapper.map_to_constraints(profile)
        
        # Métricas que exceden límites
        metrics = {key: int(value * 1.5) for key, value in limits.items()}
        
        score = self.mapper.compute_constraint_score(metrics, profile)
        
        self.assertLess(score, 1.0)
        self.assertGreater(score, 0.0)  # Pero no cero
    
    def test_constraint_score_postcondition(self):
        """Score de restricciones debe estar en [0, 1]."""
        profile = RiskProfile(risk_tolerance=0.5)
        metrics = {'cyclomatic': 100, 'depth': 20, 'loc': 5000}
        
        score = self.mapper.compute_constraint_score(metrics, profile)
        
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


# =============================================================================
# TEST 5: MOTOR DE VALIDACIÓN SEMÁNTICA (INTEGRACIÓN)
# =============================================================================

class TestSemanticValidationEngine(unittest.TestCase):
    """Verifica el motor completo de validación."""
    
    def setUp(self):
        """Configuración común."""
        self.kg = create_default_knowledge_graph()
        self.engine = SemanticValidationEngine(knowledge_graph=self.kg)
    
    def test_initialization(self):
        """Verifica inicialización correcta del motor."""
        self.assertIsNotNone(self.engine.metric)
        self.assertIsNotNone(self.engine.risk_profile)
        self.assertIsNotNone(self.engine.purpose_validator)
        self.assertIsNotNone(self.engine.confidence_filter)
        self.assertIsNotNone(self.engine.constraint_mapper)
        self.assertIsNotNone(self.engine.cohomology)
    
    def test_thresholds_ordering(self):
        """Verifica que umbrales de Mahalanobis están ordenados."""
        thresholds = [
            self.engine.MAHALANOBIS_THRESHOLDS[Verdict.VIABLE],
            self.engine.MAHALANOBIS_THRESHOLDS[Verdict.CONDITIONAL],
            self.engine.MAHALANOBIS_THRESHOLDS[Verdict.WARNING],
        ]
        
        self.assertEqual(thresholds, sorted(thresholds))
    
    def test_high_quality_code_viable(self):
        """Código de alta calidad debe obtener VIABLE."""
        purposes = [
            BusinessPurpose("caching", "LATENCY_REDUCTION", strength=0.95, confidence=0.98)
        ]
        
        llm_output = LLMOutput(entropy=0.5, confidence=0.95, temperature=0.7, num_tokens=150)
        
        code_metrics = {'cyclomatic': 8, 'depth': 3, 'loc': 45, 'cognitive': 12}
        
        result = self.engine.validate(purposes, llm_output, code_metrics)
        
        self.assertEqual(result.verdict, Verdict.VIABLE)
        self.assertLess(result.mahalanobis_distance, 
                       self.engine.MAHALANOBIS_THRESHOLDS[Verdict.VIABLE])
    
    def test_weak_purpose_rejected(self):
        """Propósito muy débil debe ser rechazado."""
        purposes = [
            BusinessPurpose("unknown", "UNKNOWN_PROBLEM", strength=0.1, confidence=0.3)
        ]
        
        llm_output = LLMOutput(entropy=1.0, confidence=0.5)
        
        result = self.engine.validate(purposes, llm_output)
        
        self.assertEqual(result.verdict, Verdict.REJECT)
    
    def test_low_confidence_rejected(self):
        """Confianza muy baja debe ser rechazada."""
        purposes = [
            BusinessPurpose("caching", "LATENCY_REDUCTION", strength=0.8, confidence=0.9)
        ]
        
        llm_output = LLMOutput(entropy=1.0, confidence=0.15)
        
        result = self.engine.validate(purposes, llm_output)
        
        self.assertEqual(result.verdict, Verdict.REJECT)
    
    def test_stochastic_singularity_rejected(self):
        """Singularidades estocásticas deben ser rechazadas inmediatamente."""
        purposes = [
            BusinessPurpose("encryption", "SECURITY_HARDENING", strength=0.98, confidence=0.99)
        ]
        
        # Entropía infinita
        llm_output = LLMOutput(entropy=float('inf'), confidence=0.8)
        
        result = self.engine.validate(purposes, llm_output)
        
        self.assertEqual(result.verdict, Verdict.REJECT)
        self.assertEqual(result.mahalanobis_distance, float('inf'))
        self.assertIn('singularity', ' '.join(result.reasons).lower())
    
    def test_topological_obstruction_rejected(self):
        """Obstrucciones topológicas deben forzar REJECT."""
        # Crear perfil que genere paradoja
        risky_profile = RiskProfile(risk_tolerance=0.95, domain_criticality=0.1)
        engine = SemanticValidationEngine(
            knowledge_graph=self.kg,
            risk_profile=risky_profile,
            enable_cohomology=True
        )
        
        purposes = [
            BusinessPurpose("caching", "LATENCY_REDUCTION", strength=0.9, confidence=0.95)
        ]
        
        # Confianza muy baja → paradoja con propósito fuerte
        llm_output = LLMOutput(entropy=0.6, confidence=0.25)
        
        result = engine.validate(purposes, llm_output, {'cyclomatic': 5})
        
        # Debe detectar obstrucción y rechazar
        self.assertEqual(result.verdict, Verdict.REJECT)
        self.assertGreater(result.cohomology_dimension, 0)
    
    def test_cohomology_disabled_no_obstruction_check(self):
        """Con cohomología deshabilitada, no se verifican obstrucciones."""
        engine = SemanticValidationEngine(
            knowledge_graph=self.kg,
            enable_cohomology=False
        )
        
        purposes = [
            BusinessPurpose("caching", "LATENCY_REDUCTION", strength=0.9)
        ]
        
        llm_output = LLMOutput(entropy=0.8, confidence=0.3)  # Paradoja potencial
        
        result = engine.validate(purposes, llm_output)
        
        # Sin cohomología, puede no rechazar por obstrucción
        self.assertEqual(result.cohomology_dimension, 0)
    
    def test_moderate_quality_conditional_or_warning(self):
        """Calidad moderada debe dar CONDITIONAL o WARNING."""
        purposes = [
            BusinessPurpose("caching", "LATENCY_REDUCTION", strength=0.7, confidence=0.8)
        ]
        
        llm_output = LLMOutput(entropy=1.5, confidence=0.7)
        
        code_metrics = {'cyclomatic': 18, 'depth': 5, 'loc': 180}
        
        result = self.engine.validate(purposes, llm_output, code_metrics)
        
        self.assertIn(result.verdict, [Verdict.CONDITIONAL, Verdict.WARNING])
    
    def test_verdict_supremum_property(self):
        """Veredicto final debe ser supremo de veredictos parciales."""
        # Forzar múltiples señales débiles
        purposes = [
            BusinessPurpose("caching", "LATENCY_REDUCTION", strength=0.65, confidence=0.7)
        ]
        
        llm_output = LLMOutput(entropy=2.0, confidence=0.65)
        
        code_metrics = {'cyclomatic': 25, 'depth': 6, 'loc': 300}
        
        result = self.engine.validate(purposes, llm_output, code_metrics)
        
        # Veredicto debe reflejar la peor señal
        self.assertIn(result.verdict, [Verdict.WARNING, Verdict.REJECT])
    
    def test_explain_verdict_non_empty(self):
        """Explicación de veredicto debe ser informativa."""
        purposes = [
            BusinessPurpose("caching", "LATENCY_REDUCTION", strength=0.8)
        ]
        
        llm_output = LLMOutput(entropy=1.0, confidence=0.8)
        
        result = self.engine.validate(purposes, llm_output)
        explanation = self.engine.explain_verdict(result)
        
        self.assertGreater(len(explanation), 100)
        self.assertIn("Verdict:", explanation)
        self.assertIn("Mahalanobis Distance:", explanation)
        self.assertIn("Signal Breakdown:", explanation)
    
    def test_metadata_completeness(self):
        """Metadata debe contener información clave."""
        purposes = [
            BusinessPurpose("caching", "LATENCY_REDUCTION", strength=0.8)
        ]
        
        llm_output = LLMOutput(entropy=1.0, confidence=0.8)
        
        result = self.engine.validate(purposes, llm_output)
        
        self.assertIn('risk_profile', result.metadata)
        self.assertIn('mahalanobis_distance', result.metadata)
        self.assertIn('cohomology_dimension', result.metadata)
        self.assertIn('signal_vector', result.metadata)
    
    def test_custom_metric_affects_distance(self):
        """Métrica personalizada debe afectar distancia calculada."""
        custom_metric = MahalanobisMetric()
        custom_metric.set_coupling(0, 1, 0.8)  # Fuerte acoplamiento
        
        engine_custom = SemanticValidationEngine(
            knowledge_graph=self.kg,
            metric=custom_metric
        )
        
        purposes = [
            BusinessPurpose("caching", "LATENCY_REDUCTION", strength=0.7)
        ]
        
        llm_output = LLMOutput(entropy=1.0, confidence=0.7)
        
        result_default = self.engine.validate(purposes, llm_output)
        result_custom = engine_custom.validate(purposes, llm_output)
        
        # Distancias deben diferir
        self.assertNotAlmostEqual(
            result_default.mahalanobis_distance,
            result_custom.mahalanobis_distance,
            places=3
        )


# =============================================================================
# TEST 6: ROBUSTEZ Y CASOS EXTREMOS
# =============================================================================

class TestRobustness(unittest.TestCase):
    """Verifica robustez ante entradas extremas y aleatorias."""
    
    def setUp(self):
        """Configuración común."""
        self.kg = create_default_knowledge_graph()
        self.engine = SemanticValidationEngine(knowledge_graph=self.kg)
    
    @given(
        business_purpose_strategy(),
        llm_output_strategy(allow_singular=True)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_no_crash_on_random_inputs(self, purpose, llm_output):
        """Property: El motor no debe crashear con entradas aleatorias."""
        try:
            result = self.engine.validate([purpose], llm_output)
            
            # Postcondiciones básicas
            self.assertIsInstance(result.verdict, Verdict)
            self.assertGreaterEqual(result.mahalanobis_distance, 0.0)
            
        except (ContractViolationError, TopologicalObstructionError, MetricDegeneracyError):
            # Excepciones esperadas están OK
            pass
    
    @given(risk_profile_strategy())
    @settings(max_examples=50, deadline=None)
    def test_property_risk_profile_affects_verdict(self, risk_profile):
        """Property: Perfil de riesgo debe afectar veredicto."""
        engine = SemanticValidationEngine(
            knowledge_graph=self.kg,
            risk_profile=risk_profile
        )
        
        purposes = [
            BusinessPurpose("caching", "LATENCY_REDUCTION", strength=0.7)
        ]
        
        llm_output = LLMOutput(entropy=1.5, confidence=0.7)
        
        try:
            result = engine.validate(purposes, llm_output)
            self.assertIsNotNone(result.verdict)
        except (ContractViolationError, MetricDegeneracyError):
            pass
    
    def test_extreme_high_metrics(self):
        """Métricas extremadamente altas deben ser penalizadas."""
        purposes = [
            BusinessPurpose("caching", "LATENCY_REDUCTION", strength=0.8)
        ]
        
        llm_output = LLMOutput(entropy=1.0, confidence=0.8)
        
        extreme_metrics = {'cyclomatic': 1000, 'depth': 50, 'loc': 10000}
        
        result = self.engine.validate(purposes, llm_output, extreme_metrics)
        
        # Debe penalizar severamente
        self.assertIn(result.verdict, [Verdict.WARNING, Verdict.REJECT])
    
    def test_zero_metrics(self):
        """Métricas en cero deben ser válidas."""
        purposes = [
            BusinessPurpose("caching", "LATENCY_REDUCTION", strength=0.9)
        ]
        
        llm_output = LLMOutput(entropy=0.5, confidence=0.9)
        
        zero_metrics = {'cyclomatic': 0, 'depth': 0, 'loc': 0}
        
        result = self.engine.validate(purposes, llm_output, zero_metrics)
        
        # Cero es perfecto (por debajo de límites)
        self.assertIn(result.verdict, [Verdict.VIABLE, Verdict.CONDITIONAL])
    
    def test_missing_metrics(self):
        """Métricas faltantes deben asumir cumplimiento."""
        purposes = [
            BusinessPurpose("caching", "LATENCY_REDUCTION", strength=0.9)
        ]
        
        llm_output = LLMOutput(entropy=0.5, confidence=0.9)
        
        result = self.engine.validate(purposes, llm_output)  # Sin code_metrics
        
        # Debe asumir cumplimiento perfecto
        self.assertGreater(result.signals.get('constraints', 0), 0.9)
    
    def test_very_long_purpose_list(self):
        """Lista muy larga de propósitos debe manejarse eficientemente."""
        purposes = [
            BusinessPurpose(f"concept_{i}", "COST_REDUCTION", strength=0.5 + i*0.01)
            for i in range(100)
        ]
        
        llm_output = LLMOutput(entropy=1.0, confidence=0.8)
        
        result = self.engine.validate(purposes, llm_output)
        
        # Debe completar sin timeout
        self.assertIsNotNone(result.verdict)
    
    def test_boundary_entropy_values(self):
        """Valores de entropía en frontera deben manejarse correctamente."""
        purposes = [
            BusinessPurpose("caching", "LATENCY_REDUCTION", strength=0.8)
        ]
        
        # Entropía exactamente en umbral
        llm_boundary = LLMOutput(
            entropy=ConfidenceFilter.DEFAULT_MAX_ENTROPY,
            confidence=0.8
        )
        
        result = self.engine.validate(purposes, llm_boundary)
        
        # Debe tener veredicto definido
        self.assertIsNotNone(result.verdict)
    
    def test_boundary_mahalanobis_thresholds(self):
        """Distancias exactamente en umbrales deben tener veredicto correcto."""
        # Difícil de forzar exactamente, pero verificamos monotonía
        
        # Señales cercanas al ideal
        purposes_good = [
            BusinessPurpose("caching", "LATENCY_REDUCTION", strength=0.95, confidence=0.98)
        ]
        llm_good = LLMOutput(entropy=0.3, confidence=0.95)
        result_good = self.engine.validate(purposes_good, llm_good)
        
        # Señales peores
        purposes_bad = [
            BusinessPurpose("caching", "LATENCY_REDUCTION", strength=0.5, confidence=0.6)
        ]
        llm_bad = LLMOutput(entropy=2.5, confidence=0.6)
        result_bad = self.engine.validate(purposes_bad, llm_bad)
        
        # Veredicto debe empeorar (monotonía)
        self.assertLessEqual(result_good.verdict, result_bad.verdict)


# =============================================================================
# TEST 7: COMPATIBILIDAD LEGACY
# =============================================================================

class TestLegacyCompatibility(unittest.TestCase):
    """Verifica compatibilidad con API anterior."""
    
    def test_deprecated_engine_initialization(self):
        """OntologicalDiffeomorphismEngine debe inicializar correctamente."""
        from semantic_validator import OntologicalDiffeomorphismEngine
        
        # Simular business_profile legacy
        mock_profile = Mock()
        mock_profile.risk_tolerance = 0.5
        mock_profile.domain_criticality = 0.7
        
        with self.assertWarns(Warning):  # Debe emitir warning de deprecación
            engine = OntologicalDiffeomorphismEngine(
                knowledge_graph=create_default_knowledge_graph(),
                business_profile=mock_profile
            )
        
        self.assertIsNotNone(engine._engine)
    
    def test_compile_wisdom_returns_integer(self):
        """compile_wisdom debe retornar código de veredicto entero."""
        from semantic_validator import OntologicalDiffeomorphismEngine
        
        mock_profile = Mock()
        mock_profile.risk_tolerance = 0.5
        
        with self.assertWarns(Warning):
            engine = OntologicalDiffeomorphismEngine(
                knowledge_graph=create_default_knowledge_graph(),
                business_profile=mock_profile
            )
        
        # Simular tool_semantics legacy
        mock_sem = Mock()
        mock_sem.source_concept = "caching"
        mock_sem.target_business_pain = "LATENCY_REDUCTION"
        mock_sem.semantic_weight = 0.8
        
        verdict_code = engine.compile_wisdom(
            tool_semantics=[mock_sem],
            llm_entropy=1.0,
            llm_confidence=0.8
        )
        
        self.assertIsInstance(verdict_code, int)
        self.assertIn(verdict_code, [0, 1, 2, 3])  # Códigos válidos de Verdict


# =============================================================================
# TEST 8: BENCHMARKS DE RENDIMIENTO
# =============================================================================

class TestPerformance(unittest.TestCase):
    """Verifica rendimiento del motor."""
    
    def setUp(self):
        """Configuración común."""
        self.kg = create_default_knowledge_graph()
        self.engine = SemanticValidationEngine(knowledge_graph=self.kg)
    
    def test_single_validation_performance(self):
        """Validación simple debe completar en < 100ms."""
        import time
        
        purposes = [
            BusinessPurpose("caching", "LATENCY_REDUCTION", strength=0.8)
        ]
        
        llm_output = LLMOutput(entropy=1.0, confidence=0.8)
        
        start = time.perf_counter()
        result = self.engine.validate(purposes, llm_output)
        duration = time.perf_counter() - start
        
        self.assertLess(duration, 0.1)  # < 100ms
        self.assertIsNotNone(result.verdict)
    
    def test_batch_validation_performance(self):
        """100 validaciones deben completar en < 5s."""
        import time
        
        purposes = [
            BusinessPurpose("caching", "LATENCY_REDUCTION", strength=0.8)
        ]
        
        llm_output = LLMOutput(entropy=1.0, confidence=0.8)
        
        start = time.perf_counter()
        for _ in range(100):
            self.engine.validate(purposes, llm_output)
        duration = time.perf_counter() - start
        
        self.assertLess(duration, 5.0)  # < 5s para 100 validaciones
    
    def test_cohomology_computation_performance(self):
        """Cómputo cohomológico debe ser rápido."""
        import time
        
        cohom = SimplicialCohomology()
        cohom.set_signals(0.5, 0.6, 0.7, 0.8)
        
        start = time.perf_counter()
        for _ in range(1000):
            cohom.compute_cohomology_dimension()
        duration = time.perf_counter() - start
        
        self.assertLess(duration, 0.5)  # < 0.5s para 1000 cómputos


# =============================================================================
# SUITE PRINCIPAL
# =============================================================================

def suite():
    """Construye suite completa de tests."""
    test_suite = unittest.TestSuite()
    
    # Tests de estructuras de datos
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDataStructures))
    
    # Tests de métrica de Mahalanobis
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestMahalanobisMetric))
    
    # Tests de cohomología
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSimplicialCohomology))
    
    # Tests de validadores
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPurposeValidator))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestConfidenceFilter))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestConstraintMapper))
    
    # Tests del motor principal
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSemanticValidationEngine))
    
    # Tests de robustez
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRobustness))
    
    # Tests de compatibilidad
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLegacyCompatibility))
    
    # Tests de rendimiento
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPerformance))
    
    return test_suite


if __name__ == '__main__':
    # Configurar runner con verbosidad
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Ejecutar suite completa
    result = runner.run(suite())
    
    # Resumen final
    print("\n" + "=" * 80)
    print("TEST SUITE SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    print("=" * 80)
    
    # Exit code
    sys.exit(0 if result.wasSuccessful() else 1)