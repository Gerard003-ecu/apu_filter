"""
=========================================================================================
Test Suite: Semantic Dictionary - Validación Rigurosa de Invariantes Matemáticos
Ubicación: tests/test_semantic_dictionary.py
=========================================================================================

METODOLOGÍA DE TESTING:

1. **Property-Based Testing**:
   Usa Hypothesis para generar casos automáticos que verifican propiedades
   matemáticas universales (ej: β₀ ≥ 1, eigenvalues reales para matrices simétricas).

2. **Verificación de Invariantes Topológicos**:
   - Fórmula de Euler: V - E + F = χ
   - Monotonía de números de Betti en filtraciones
   - Simetría y positividad semidefinida del Laplaciano

3. **Análisis de Complejidad**:
   Benchmarks para verificar que los algoritmos cumplen con su complejidad teórica.

4. **Fuzzing Matemático**:
   Generación de grafos aleatorios para detectar casos extremos no contemplados.

=========================================================================================
"""

import hashlib
import logging
import random
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as npst
from unittest.mock import Mock, patch, MagicMock

# Imports del módulo bajo test
import sys
from pathlib import Path

# Agregar el path del proyecto
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.wisdom.semantic_dictionary import (
    # Clases principales
    SemanticDictionaryService,
    GraphSemanticProjector,
    PyramidalSemanticVector,
    SemanticCache,
        TemplateValidator,
    
    # Utilidades matemáticas
            StatisticalThresholdClassifier,
    
    # Constantes y tipos
    Stratum,
    NodeType,
    VALID_NODE_TYPES,
    EPSILON_SPECTRAL,
    EPSILON_TOPOLOGY,
    
    # Factory
    create_semantic_dictionary_service,
)


# =============================================================================
# CONFIGURACIÓN DE LOGGING PARA TESTS
# =============================================================================

logging.basicConfig(
    level=logging.WARNING,  # Solo warnings y errors en tests
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("test_semantic_dictionary")


# =============================================================================
# FIXTURES COMPARTIDOS
# =============================================================================

@pytest.fixture
def simple_adjacency_matrix() -> np.ndarray:
    """
    Grafo simple conexo sin ciclos (árbol):
    
        0 --- 1 --- 2
              |
              3
    
    Propiedades:
        - V = 4 vértices
        - E = 3 aristas
        - β₀ = 1 (conexo)
        - β₁ = 0 (acíclico)
        - χ = V - E = 1
    """
    adj = np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 0]
    ], dtype=float)
    return adj


@pytest.fixture
def cycle_adjacency_matrix() -> np.ndarray:
    """
    Grafo con un ciclo:
    
        0 --- 1
        |     |
        3 --- 2
    
    Propiedades:
        - V = 4 vértices
        - E = 4 aristas
        - β₀ = 1 (conexo)
        - β₁ = 1 (un ciclo)
        - χ = V - E = 0
    """
    adj = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ], dtype=float)
    return adj


@pytest.fixture
def disconnected_adjacency_matrix() -> np.ndarray:
    """
    Grafo con dos componentes desconectadas:
    
        0 --- 1        2 --- 3
    
    Propiedades:
        - V = 4 vértices
        - E = 2 aristas
        - β₀ = 2 (dos componentes)
        - β₁ = 0 (acíclico)
        - χ = V - E = 2
    """
    adj = np.array([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=float)
    return adj


@pytest.fixture
def laplacian_matrix(simple_adjacency_matrix) -> np.ndarray:
    """
    Matriz Laplaciana del grafo simple.
    
    L = D - A, donde D es la matriz de grados.
    """
    adj = simple_adjacency_matrix
    degrees = np.sum(adj, axis=1)
    D = np.diag(degrees)
    L = D - adj
    return L


@pytest.fixture
def semantic_vector() -> PyramidalSemanticVector:
    """Vector semántico de prueba."""
    return PyramidalSemanticVector(
        node_id="APU_001",
        node_type="APU",
        stratum=Stratum.TACTICS,
        in_degree=3,
        out_degree=5,
        is_critical_bridge=True,
        weight=100.0
    )


@pytest.fixture
def semantic_service() -> SemanticDictionaryService:
    """Servicio semántico configurado para testing."""
    return create_semantic_dictionary_service(
        enable_validation=True,
        enable_statistical=False
    )


# =============================================================================
# TESTS DE UTILIDADES MATEMÁTICAS
# =============================================================================

class TestStatisticalThresholdClassifier:
    """Tests para clasificador estadístico."""
    
    def test_classifier_fit_and_classify(self):
        """
        Test: Clasificador ajusta umbrales correctamente.
        """
        data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        
        classifier = StatisticalThresholdClassifier(
            metric_name="test_metric",
            quantiles={
                "low": 0.25,
                "medium": 0.50,
                "high": 0.75,
            }
        )
        
        classifier.fit(data)
        
        # Verificar umbrales
        assert classifier._thresholds["low"] == 32.5  # Q1
        assert classifier._thresholds["medium"] == 55.0  # Mediana
        assert classifier._thresholds["high"] == 77.5  # Q3
        
        # Clasificar valores
        assert classifier.classify(20) == "low"
        assert classifier.classify(50) == "medium"
        assert classifier.classify(85) == "high"
    
    def test_classifier_not_fitted_raises(self):
        """
        Test: Clasificador no ajustado lanza error.
        """
        classifier = StatisticalThresholdClassifier(
            metric_name="test",
            quantiles={"low": 0.5}
        )
        
        with pytest.raises(ValueError, match="not fitted"):
            classifier.classify(10.0)
    
    def test_confidence_interval_bootstrap(self):
        """
        Test: Intervalos de confianza son razonables.
        """
        # Datos normales
        np.random.seed(42)
        data = np.random.normal(loc=50, scale=10, size=100)
        
        classifier = StatisticalThresholdClassifier(
            metric_name="test",
            quantiles={"median": 0.5},
            reference_distribution=data
        )
        
        classifier.fit(data)
        
        ci = classifier.get_confidence_interval("median", confidence=0.95)
        
        assert ci is not None, "CI should be computed"
        lower, upper = ci
        assert lower < upper, "Lower bound must be less than upper"
        assert lower < 50 < upper, "True median should be in CI"


class TestGiniCoefficient:
    """Tests para coeficiente de Gini."""
    
    def test_gini_perfect_equality(self):
        """
        Test: Gini = 0 para distribución perfectamente igual.
        """
        values = np.array([10, 10, 10, 10, 10])
        gini = GraphSemanticProjector._gini_coefficient(values)
        
        assert abs(gini) < EPSILON_TOPOLOGY, (
            f"Gini should be 0 for perfect equality, got {gini}"
        )
    
    def test_gini_maximum_inequality(self):
        """
        Test: Gini → 1 para máxima desigualdad.
        """
        # Uno tiene todo, los demás nada
        values = np.array([0, 0, 0, 0, 100])
        gini = GraphSemanticProjector._gini_coefficient(values)
        
        # Para n elementos, Gini máximo es (n-1)/n
        expected_max = 4/5  # 0.8 para n=5
        
        assert gini >= expected_max * 0.9, (
            f"Gini should be close to {expected_max} for max inequality, "
            f"got {gini}"
        )
    
    def test_gini_empty_array(self):
        """
        Test: Gini = 0 para array vacío.
        """
        values = np.array([])
        gini = GraphSemanticProjector._gini_coefficient(values)
        
        assert gini == 0.0
    
    def test_gini_single_value(self):
        """
        Test: Gini = 0 para un solo valor.
        """
        values = np.array([42])
        gini = GraphSemanticProjector._gini_coefficient(values)
        
        assert gini == 0.0
    
    def test_gini_properties(self):
        """
        Test: Propiedades matemáticas del Gini.
        
        1. 0 ≤ Gini ≤ 1
        2. Invariante ante escalamiento
        3. Aumenta con desigualdad
        """
        # Distribución moderadamente desigual
        values = np.array([1, 2, 3, 4, 5])
        gini1 = GraphSemanticProjector._gini_coefficient(values)
        
        # Propiedad 1: Rango
        assert 0 <= gini1 <= 1, "Gini must be in [0, 1]"
        
        # Propiedad 2: Invariancia ante escalamiento
        scaled_values = values * 10
        gini2 = GraphSemanticProjector._gini_coefficient(scaled_values)
        assert abs(gini1 - gini2) < EPSILON_TOPOLOGY, (
            "Gini must be scale-invariant"
        )
        
        # Propiedad 3: Aumenta con desigualdad
        more_unequal = np.array([1, 1, 1, 1, 10])
        gini3 = GraphSemanticProjector._gini_coefficient(more_unequal)
        assert gini3 > gini1, "Gini should increase with inequality"


# =============================================================================
# TESTS DE CACHÉ TTL
# =============================================================================

class TestPyramidalSemanticVector:
    """Tests para vectores semánticos."""
    
    def test_vector_creation_valid(self, semantic_vector):
        """
        Test: Creación de vector con parámetros válidos.
        """
        assert semantic_vector.node_id == "APU_001"
        assert semantic_vector.node_type == "APU"
        assert semantic_vector.stratum == Stratum.TACTICS
        assert semantic_vector.total_degree == 8
    
    def test_vector_negative_degree_raises(self):
        """
        Test: Grados negativos lanzan error.
        """
        with pytest.raises(ValueError, match="non-negative"):
            PyramidalSemanticVector(
                node_id="test",
                node_type="APU",
                stratum=Stratum.TACTICS,
                in_degree=-1,
                out_degree=5
            )
    
    def test_vector_invalid_node_type_raises(self):
        """
        Test: Tipo de nodo inválido lanza error.
        """
        with pytest.raises(ValueError, match="Invalid node_type"):
            PyramidalSemanticVector(
                node_id="test",
                node_type="INVALID_TYPE",  # type: ignore
                stratum=Stratum.TACTICS,
                in_degree=1,
                out_degree=1
            )
    
    def test_vector_empty_node_id_raises(self):
        """
        Test: node_id vacío lanza error.
        """
        with pytest.raises(ValueError, match="cannot be empty"):
            PyramidalSemanticVector(
                node_id="   ",
                node_type="APU",
                stratum=Stratum.TACTICS,
                in_degree=1,
                out_degree=1
            )
    
    def test_vector_properties(self, semantic_vector):
        """
        Test: Propiedades derivadas del vector.
        """
        assert semantic_vector.total_degree == 8
        assert not semantic_vector.is_leaf  # out_degree > 0
        assert not semantic_vector.is_root  # in_degree > 0
        assert not semantic_vector.is_isolated  # total_degree > 0
        
        # Vector aislado
        isolated = PyramidalSemanticVector(
            node_id="isolated",
            node_type="INSUMO",
            stratum=Stratum.PHYSICS,
            in_degree=0,
            out_degree=0
        )
        
        assert isolated.is_isolated
        assert isolated.is_leaf
        assert isolated.is_root
    
    def test_vector_to_dict(self, semantic_vector):
        """
        Test: Serialización a diccionario.
        """
        data = semantic_vector.to_dict()
        
        assert data["node_id"] == "APU_001"
        assert data["total_degree"] == 8
        assert data["is_critical_bridge"] is True
        assert "stratum" in data
    
    def test_vector_with_updates(self, semantic_vector):
        """
        Test: Inmutabilidad funcional con with_updates.
        """
        updated = semantic_vector.with_updates(in_degree=10)
        
        assert updated.in_degree == 10
        assert semantic_vector.in_degree == 3  # Original no mutado
        assert updated.node_id == semantic_vector.node_id
    
class TestGraphSemanticProjector:
    """Tests para proyector semántico."""
    
    @pytest.fixture
    def mock_dictionary(self):
        """Mock del servicio de diccionario."""
        mock = Mock(spec=SemanticDictionaryService)
        mock.fetch_narrative.return_value = {
            "success": True,
            "narrative": "Test narrative"
        }
        return mock
    
    @pytest.fixture
    def projector(self, mock_dictionary):
        """Proyector con diccionario mock."""
        return GraphSemanticProjector(
            dictionary_service=mock_dictionary,
            cache_ttl=60,
            cache_maxsize=100
        )
    
    def test_projector_stress_point(self, projector, semantic_vector, mock_dictionary):
        """
        Test: Proyección de punto de estrés.
        """
        result = projector.project_pyramidal_stress(semantic_vector)
        
        assert "vector_metadata" in result
        assert "criticality_score" in result
        assert result["vector_metadata"]["node_id"] == "APU_001"
        
        # Verificar que llamó al diccionario
        mock_dictionary.fetch_narrative.assert_called_once()
    
    def test_projector_stress_caching(self, projector, semantic_vector):
        """
        Test: Caché evita llamadas duplicadas.
        """
        # Primera llamada
        result1 = projector.project_pyramidal_stress(semantic_vector)
        
        # Segunda llamada (debería venir del caché)
        result2 = projector.project_pyramidal_stress(semantic_vector)
        
        # Verificar que el diccionario solo se llamó una vez
        assert projector._dictionary.fetch_narrative.call_count == 1
        
        # Resultados deben ser iguales
        assert result1["vector_metadata"] == result2["vector_metadata"]
    
    def test_projector_criticality_score(self, projector):
        """
        Test: Cálculo de score de criticidad.
        """
        # Nodo altamente crítico
        critical_vector = PyramidalSemanticVector(
            node_id="critical",
            node_type="APU",
            stratum=Stratum.TACTICS,
            in_degree=50,
            out_degree=50,
            is_critical_bridge=True
        )
        
        score = projector._compute_criticality(critical_vector)
        
        assert 0 <= score <= 1, "Score must be in [0, 1]"
        assert score > 0.5, "High degree + bridge should have high score"
        
        # Nodo poco crítico
        low_vector = PyramidalSemanticVector(
            node_id="low",
            node_type="INSUMO",
            stratum=Stratum.PHYSICS,
            in_degree=1,
            out_degree=0,
            is_critical_bridge=False
        )
        
        low_score = projector._compute_criticality(low_vector)
        assert low_score < 0.5, "Low degree + not bridge should have low score"
    
    def test_projector_cycle_path(self, projector):
        """
        Test: Proyección de ciclo.
        """
        path = ["APU_001", "INSUMO_042", "APU_003"]
        
        result = projector.project_cycle_path(path)
        
        assert "cycle_metadata" in result
        assert result["cycle_metadata"]["length"] == 3
        assert not result["cycle_metadata"]["is_self_loop"]
    
    def test_projector_self_loop(self, projector):
        """
        Test: Proyección de self-loop.
        """
        path = ["APU_001"]
        
        result = projector.project_cycle_path(path)
        
        assert result["cycle_metadata"]["is_self_loop"]
        assert result["cycle_metadata"]["length"] == 1
    
    def test_projector_empty_cycle_returns_error(self, projector):
        """
        Test: Ciclo vacío retorna error.
        """
        result = projector.project_cycle_path([])
        
        assert result["success"] is False
        assert "error" in result
    
    def test_projector_fragmentation(self, projector):
        """
        Test: Proyección de fragmentación.
        """
        component_sizes = [10, 5, 3, 2]
        
        result = projector.project_fragmentation(
            beta_0=4,
            component_sizes=component_sizes
        )
        
        assert "component_analysis" in result
        assert result["component_analysis"]["largest"] == 10
        assert result["component_analysis"]["smallest"] == 2
        assert "gini_coefficient" in result["component_analysis"]
    
    def test_projector_fragmentation_classification(self, projector):
        """
        Test: Clasificación correcta según β₀.
        """
        # β₀ = 0: empty
        result = projector.project_fragmentation(beta_0=0)
        assert projector._dictionary.fetch_narrative.call_args[1]["classification"] == "empty"
        
        # β₀ = 1: unified
        projector._dictionary.fetch_narrative.reset_mock()
        result = projector.project_fragmentation(beta_0=1)
        assert projector._dictionary.fetch_narrative.call_args[1]["classification"] == "unified"
        
        # β₀ > 5: severely_fragmented
        projector._dictionary.fetch_narrative.reset_mock()
        result = projector.project_fragmentation(beta_0=10)
        assert projector._dictionary.fetch_narrative.call_args[1]["classification"] == "severely_fragmented"
    
    def test_projector_cache_stats(self, projector, semantic_vector):
        """
        Test: Estadísticas del caché del proyector.
        """
        # Generar algunas proyecciones
        projector.project_pyramidal_stress(semantic_vector)
        projector.project_cycle_path(["A", "B", "C"])
        
        stats = projector.cache_stats
        
        assert "size" in stats
        assert "hits" in stats
        assert "misses" in stats
    
    def test_projector_shutdown(self, projector):
        """
        Test: Shutdown libera recursos.
        """
        projector.shutdown()
        # No debería lanzar errores


# =============================================================================
# TESTS DE VALIDADOR DE PLANTILLAS
# =============================================================================

class TestTemplateValidator:
    """Tests para validador de plantillas."""
    
    def test_extract_placeholders_simple(self):
        """
        Test: Extracción de placeholders simples.
        """
        template = "Hello {name}, you have {count} messages."
        placeholders = TemplateValidator.extract_placeholders(template)
        
        assert placeholders == {"name", "count"}
    
    def test_extract_placeholders_with_format(self):
        """
        Test: Extracción con especificadores de formato.
        """
        template = "Value: {value:.2f}, Percent: {pct:.1%}"
        placeholders = TemplateValidator.extract_placeholders(template)
        
        assert placeholders == {"value", "pct"}
    
    def test_extract_placeholders_nested(self):
        """
        Test: Extracción de placeholders con atributos.
        """
        template = "User {user.name} has {user.score} points"
        placeholders = TemplateValidator.extract_placeholders(template)
        
        assert "user" in placeholders
    
    def test_validate_template_valid(self):
        """
        Test: Validación de plantilla válida.
        """
        template = "Result: {beta_1} cycles detected"
        is_valid, error = TemplateValidator.validate_template(template)
        
        assert is_valid
        assert error is None
    
    def test_validate_template_invalid_syntax(self):
        """
        Test: Plantilla con sintaxis inválida.
        """
        template = "Result: {unclosed"
        is_valid, error = TemplateValidator.validate_template(template)
        
        assert not is_valid
        assert error is not None
    
    def test_validate_template_with_required_params(self):
        """
        Test: Validación con parámetros requeridos.
        """
        template = "Value: {x}"
        
        # Falta parámetro 'y'
        is_valid, error = TemplateValidator.validate_template(
            template,
            required_params={"x", "y"}
        )
        
        assert not is_valid
        assert "Missing required parameters" in error
    
    def test_validate_all_templates(self):
        """
        Test: Validación recursiva de plantillas anidadas.
        """
        templates = {
            "level1": {
                "valid": "Test {param}",
                "invalid": "Test {unclosed",
            },
            "level2": "Another {value:.2f}"
        }
        
        errors = TemplateValidator.validate_all_templates(templates)
        
        assert len(errors) == 1  # Solo "invalid" tiene error
        assert errors[0]["path"] == "level1.invalid"
    
    def test_infer_test_value_float(self):
        """
        Test: Inferencia de valor de prueba para floats.
        """
        value = TemplateValidator._infer_test_value(".2f")
        assert isinstance(value, float)
        
        value = TemplateValidator._infer_test_value(".3e")
        assert isinstance(value, float)
    
    def test_infer_test_value_int(self):
        """
        Test: Inferencia de valor de prueba para integers.
        """
        value = TemplateValidator._infer_test_value("d")
        assert isinstance(value, int)


# =============================================================================
# TESTS DEL SERVICIO PRINCIPAL
# =============================================================================


    def test_enforce_lipschitz_boundary(self):
        """
        Test: Retracto de deformación de Lipschitz.
        """
        text = "El proyecto es altamente viable y estable."

        # Con Psi > 1.0 (Estable), no debería truncar
        result_stable = TemplateValidator.enforce_lipschitz_boundary(text, 1.2)
        assert result_stable == text

        # Con Psi < 1.0 (Inestable), debería truncar palabras optimistas
        result_unstable = TemplateValidator.enforce_lipschitz_boundary(text, 0.8)
        assert "⚠️ ALERTA DE SISTEMA" in result_unstable
        assert "PRECAUCIÓN/RECHAZO" in result_unstable


class TestSemanticDictionaryService:
    """Tests para el servicio principal."""
    
    def test_service_initialization(self, semantic_service):
        """
        Test: Inicialización correcta del servicio.
        """
        assert semantic_service is not None
        assert len(semantic_service.get_available_domains()) > 0
    
    def test_service_fetch_narrative_topology(self, semantic_service):
        """
        Test: Fetch narrativa de topología.
        """
        result = semantic_service.fetch_narrative(
            domain="TOPOLOGY_CYCLES",
            classification="clean"
        )
        
        assert result["success"] is True
        assert "narrative" in result
        assert result["stratum"] == Stratum.WISDOM.name
    
    def test_service_fetch_narrative_with_params(self, semantic_service):
        """
        Test: Fetch con parámetros de sustitución.
        """
        result = semantic_service.fetch_narrative(
            domain="TOPOLOGY_CYCLES",
            classification="minor",
            params={"beta_1": 3}
        )
        
        assert result["success"] is True
        assert "3" in result["narrative"]
    
    def test_service_fetch_narrative_missing_param(self, semantic_service):
        """
        Test: Error si falta parámetro requerido.
        """
        result = semantic_service.fetch_narrative(
            domain="STABILITY",
            classification="critical",
            params={}  # Falta "stability"
        )
        
        assert result["success"] is False
        assert "error" in result
    
    def test_service_fetch_market_context(self, semantic_service):
        """
        Test: Fetch de contexto de mercado.
        """
        result = semantic_service.fetch_narrative(
            domain="MARKET_CONTEXT",
            params={"deterministic": True, "index": 0}
        )
        
        assert result["success"] is True
        assert "narrative" in result
    
    def test_service_classification_by_threshold(self, semantic_service):
        """
        Test: Clasificación por umbrales.
        """
        classification = semantic_service.get_classification_by_threshold(
            metric_name="STABILITY",
            value=0.90
        )
        assert classification in ["robust", "critical"]
    
    def test_service_invalid_metric_raises(self, semantic_service):
        """
        Test: Métrica inválida lanza error.
        """
        with pytest.raises(ValueError, match="not recognized"):
            semantic_service.get_classification_by_threshold(
                metric_name="INVALID_METRIC",
                value=0.5
            )
    
    def test_service_projector_lazy_init(self, semantic_service):
        """
        Test: Proyector se inicializa lazy.
        """
        # Acceder al proyector
        projector = semantic_service.projector
        
        assert projector is not None
        assert isinstance(projector, GraphSemanticProjector)
        
        # Segunda llamada retorna la misma instancia
        projector2 = semantic_service.projector
        assert projector is projector2
    
    def test_service_health_check(self, semantic_service):
        """
        Test: Health check retorna información correcta.
        """
        health = semantic_service.health_check()
        
        assert health["status"] == "healthy"
        assert health["service"] == "SemanticDictionaryService"
        assert "template_domains" in health
        assert "thresholds" in health
        assert "timestamp" in health
    
    def test_service_get_available_domains(self, semantic_service):
        """
        Test: Listado de dominios disponibles.
        """
        domains = semantic_service.get_available_domains()
        
        assert isinstance(domains, list)
        assert "TOPOLOGY_CYCLES" in domains
        assert "STABILITY" in domains
    
    def test_service_get_domain_classifications(self, semantic_service):
        """
        Test: Obtener clasificaciones de un dominio.
        """
        classifications = semantic_service.get_domain_classifications("STABILITY")
        
        assert isinstance(classifications, list)
        assert "critical" in classifications
        assert "robust" in classifications
    
    def test_service_shutdown(self, semantic_service):
        """
        Test: Shutdown limpio del servicio.
        """
        # Inicializar proyector
        _ = semantic_service.projector
        
        # Shutdown
        semantic_service.shutdown()
        
        # No debería lanzar errores


# =============================================================================
# TESTS DE INTEGRACIÓN
# =============================================================================

class TestIntegration:
    """Tests de integración end-to-end."""
    
    def test_full_workflow_stress_point(self):
        """
        Test: Flujo completo de proyección de punto de estrés.
        """
        # 1. Crear servicio
        service = create_semantic_dictionary_service()
        
        # 2. Crear vector
        vector = PyramidalSemanticVector(
            node_id="APU_CRITICAL",
            node_type="APU",
            stratum=Stratum.TACTICS,
            in_degree=25,
            out_degree=30,
            is_critical_bridge=True
        )
        
        # 3. Proyectar
        result = service.projector.project_pyramidal_stress(vector)
        
        # 4. Verificar resultado
        assert result["success"] is True
        assert "narrative" in result
        assert "criticality_score" in result
        assert result["vector_metadata"]["node_id"] == "APU_CRITICAL"
        
        # 5. Cleanup
        service.shutdown()
    
    def test_full_workflow_cycle_detection(self):
        """
        Test: Flujo completo de detección y narrativa de ciclo.
        """
        service = create_semantic_dictionary_service()
        
        # Ciclo detectado
        cycle_path = ["APU_A", "INSUMO_X", "APU_B", "INSUMO_Y"]
        
        result = service.projector.project_cycle_path(
            path_nodes=cycle_path,
            cycle_metadata={"detection_algorithm": "tarjan"}
        )
        
        assert result["success"] is True
        assert "cycle_metadata" in result
        assert result["cycle_metadata"]["length"] == 4
        assert "homology_obstruction" in result
        
        service.shutdown()
    
    def test_full_workflow_fragmentation_analysis(self):
        """
        Test: Flujo completo de análisis de fragmentación.
        """
        service = create_semantic_dictionary_service()
        
        # Grafo fragmentado
        beta_0 = 3
        component_sizes = [50, 30, 20]
        
        result = service.projector.project_fragmentation(
            beta_0=beta_0,
            component_sizes=component_sizes
        )
        
        assert result["success"] is True
        assert "component_analysis" in result
        assert result["component_analysis"]["count"] == 3
        assert "gini_coefficient" in result["component_analysis"]
        assert "homology_analysis" in result
        
        service.shutdown()
    
    def test_factory_function(self):
        """
        Test: Factory function crea servicio correctamente.
        """
        service = create_semantic_dictionary_service(
            enable_validation=True,
            enable_statistical=False
        )
        
        assert isinstance(service, SemanticDictionaryService)
        
        health = service.health_check()
        assert health["status"] == "healthy"
        
        service.shutdown()


# =============================================================================
# PROPERTY-BASED TESTS (Hypothesis)
# =============================================================================

class TestPropertyBased:
    """Tests basados en propiedades con Hypothesis."""
    
    @given(st.lists(st.floats(min_value=0, max_value=1000), min_size=1, max_size=100))
    def test_gini_in_range(self, values):
        """
        Propiedad: 0 ≤ Gini ≤ 1 para cualquier distribución.
        """
        values_array = np.array(values)
        gini = GraphSemanticProjector._gini_coefficient(values_array)
        
        assert 0 <= gini <= 1, f"Gini must be in [0,1], got {gini}"
    
    @given(
        st.integers(min_value=0, max_value=50),
        st.integers(min_value=0, max_value=50)
    )
    def test_vector_total_degree(self, in_deg, out_deg):
        """
        Propiedad: total_degree = in_degree + out_degree.
        """
        vector = PyramidalSemanticVector(
            node_id="test",
            node_type="APU",
            stratum=Stratum.TACTICS,
            in_degree=in_deg,
            out_degree=out_deg
        )
        
        assert vector.total_degree == in_deg + out_deg
    
# =============================================================================
# BENCHMARKS Y TESTS DE PERFORMANCE
# =============================================================================

class TestPerformance:
    """Tests de performance y escalabilidad."""
    
    def test_gini_computation_efficiency(self):
        """
        Test: Gini se calcula eficientemente para arrays grandes.
        
        Complejidad esperada: O(n log n) por el sorting
        """
        import time
        
        n = 10000
        values = np.random.rand(n)
        
        start = time.perf_counter()
        gini = GraphSemanticProjector._gini_coefficient(values)
        elapsed = time.perf_counter() - start
        
        assert elapsed < 0.1, f"Gini computation too slow: {elapsed:.3f}s"
        assert 0 <= gini <= 1
    


# =============================================================================
# TESTS DE CASOS EXTREMOS (EDGE CASES)
# =============================================================================

class TestEdgeCases:
    """Tests de casos extremos y situaciones límite."""
    
    def test_very_large_numbers(self):
        """
        Test: Manejo de números muy grandes.
        """
        vector = PyramidalSemanticVector(
            node_id="massive",
            node_type="APU",
            stratum=Stratum.TACTICS,
            in_degree=1_000_000,
            out_degree=1_000_000
        )
        
        assert vector.total_degree == 2_000_000
        
        # Criticality score debe seguir en [0, 1]
        projector = GraphSemanticProjector(
            dictionary_service=Mock(spec=SemanticDictionaryService)
        )
        score = projector._compute_criticality(vector)
        assert 0 <= score <= 1
    
    def test_unicode_node_ids(self):
        """
        Test: Node IDs con Unicode.
        """
        vector = PyramidalSemanticVector(
            node_id="Nodo_测试_🔧",
            node_type="APU",
            stratum=Stratum.TACTICS,
            in_degree=1,
            out_degree=1
        )
        
        assert vector.node_id == "Nodo_测试_🔧"
    
# =============================================================================
# TESTS DE CONCURRENCIA
# =============================================================================

class TestConcurrency:
    """Tests de comportamiento concurrente."""
    
    def test_service_thread_safety(self):
        """
        Test: Servicio es thread-safe.
        """
        service = create_semantic_dictionary_service()
        errors = []
        
        def worker(thread_id: int):
            try:
                for i in range(20):
                    result = service.fetch_narrative(
                        domain="TOPOLOGY_CYCLES",
                        classification="clean"
                    )
                    assert result["success"] is True
            except Exception as e:
                errors.append((thread_id, e))
        
        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(10)
        ]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Thread safety violations: {errors}"
        
        service.shutdown()
    
    def test_projector_concurrent_access(self):
        """
        Test: Proyector maneja acceso concurrente.
        """
        service = create_semantic_dictionary_service()
        projector = service.projector
        
        results = []
        
        def project_stress():
            vector = PyramidalSemanticVector(
                node_id=f"node_{threading.get_ident()}",
                node_type="APU",
                stratum=Stratum.TACTICS,
                in_degree=5,
                out_degree=5
            )
            result = projector.project_pyramidal_stress(vector)
            results.append(result)
        
        threads = [
            threading.Thread(target=project_stress)
            for _ in range(20)
        ]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(results) == 20
        assert all(r["success"] for r in results)
        
        service.shutdown()


# =============================================================================
# TESTS DE REGRESIÓN
# =============================================================================

class TestRegression:
    """Tests de regresión para bugs conocidos."""
    
    def test_regression_gini_zero_values(self):
        """
        Regresión: Gini con todos valores en 0 causaba división por 0.
        """
        values = np.array([0, 0, 0, 0])
        gini = GraphSemanticProjector._gini_coefficient(values)
        
        assert gini == 0.0, "Gini of all zeros should be 0"
    
    def test_regression_cache_key_collision(self):
        """
        Regresión: Claves de caché con mismos parámetros en orden diferente.
        """
        service = create_semantic_dictionary_service()
        projector = service.projector
        
        # Dos vectores con mismos grados pero diferente orden
        v1 = PyramidalSemanticVector(
            node_id="A",
            node_type="APU",
            stratum=Stratum.TACTICS,
            in_degree=5,
            out_degree=10
        )
        
        v2 = PyramidalSemanticVector(
            node_id="B",
            node_type="APU",
            stratum=Stratum.TACTICS,
            in_degree=5,
            out_degree=10
        )
        
        r1 = projector.project_pyramidal_stress(v1)
        r2 = projector.project_pyramidal_stress(v2)
        
        # Deben tener node_ids diferentes
        assert r1["vector_metadata"]["node_id"] != r2["vector_metadata"]["node_id"]
        
        service.shutdown()


# =============================================================================
# SUITE DE TESTS COMPLETA
# =============================================================================

if __name__ == "__main__":
    """
    Ejecutar suite completa de tests.
    
    Uso:
        pytest test_semantic_dictionary.py -v
        pytest test_semantic_dictionary.py -v --benchmark-only
        pytest test_semantic_dictionary.py -v -k "property"
    """
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--strict-markers",
        "-ra",  # Show summary of all test outcomes
    ])

class TestSemanticCache:
    """Tests para SemanticCache."""

    def test_cache_basic_operations(self):
        import numpy as np
        cache = SemanticCache(maxsize=10)
        cache.set("key1", "value1", embedding=np.array([1.0, 0.0, 0.0]))
        assert cache.get("key1") == "value1"
        assert cache.get("nonexistent") is None

    def test_cache_orthogonal_eviction(self):
        import numpy as np
        decision_vector = np.array([1.0, 0.0, 0.0])
        cache = SemanticCache(maxsize=10, decision_vector=decision_vector)

        # Parallel vector, should not be evicted
        cache.set("parallel", "val1", embedding=np.array([1.0, 0.0, 0.0]))
        # Orthogonal vector, should be evicted
        cache.set("orthogonal", "val2", embedding=np.array([0.0, 1.0, 0.0]))

        # Manually cleanup orthogonal
        cache.cleanup_orthogonal()

        assert cache.get("parallel") == "val1"
        assert cache.get("orthogonal") is None
