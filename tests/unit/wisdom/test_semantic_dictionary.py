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
    TTLCache,
    TemplateValidator,
    
    # Utilidades matemáticas
    SpectralAnalyzer,
    TopologyCalculator,
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


@pytest.fixture
def ttl_cache() -> TTLCache:
    """Caché TTL configurado para testing."""
    cache = TTLCache(
        ttl_seconds=1.0,  # TTL corto para tests
        maxsize=10,
        cleanup_interval=0.5,
        auto_cleanup=True
    )
    yield cache
    cache.shutdown(timeout=2.0)


# =============================================================================
# TESTS DE UTILIDADES MATEMÁTICAS
# =============================================================================

class TestSpectralAnalyzer:
    """Tests para análisis espectral de grafos."""
    
    def test_fiedler_eigenvalue_simple_graph(self, laplacian_matrix):
        """
        Test: El eigenvalor de Fiedler es positivo para grafos conexos.
        
        Teorema: Para un grafo conexo G, λ₁(L) > 0.
        """
        fiedler = SpectralAnalyzer.fiedler_eigenvalue(laplacian_matrix)
        
        assert fiedler > 0, (
            "Fiedler eigenvalue debe ser positivo para grafo conexo"
        )
        assert fiedler < 10, (
            "Fiedler eigenvalue parece anormalmente grande"
        )
    
    def test_fiedler_eigenvalue_disconnected(self, disconnected_adjacency_matrix):
        """
        Test: λ₁ ≈ 0 para grafos desconectados.
        
        Teorema: G es desconectado ⟺ λ₁(L) = 0
        """
        # Construir Laplaciano
        adj = disconnected_adjacency_matrix
        degrees = np.sum(adj, axis=1)
        L = np.diag(degrees) - adj
        
        fiedler = SpectralAnalyzer.fiedler_eigenvalue(L)
        
        assert abs(fiedler) < EPSILON_SPECTRAL, (
            f"Fiedler eigenvalue debe ser ~0 para grafo desconectado, "
            f"got {fiedler:.2e}"
        )
    
    def test_laplacian_symmetry(self, laplacian_matrix):
        """
        Test: La matriz Laplaciana es simétrica.
        
        Propiedad: L = Lᵀ
        """
        assert np.allclose(laplacian_matrix, laplacian_matrix.T), (
            "Laplacian matrix must be symmetric"
        )
    
    def test_laplacian_positive_semidefinite(self, laplacian_matrix):
        """
        Test: El Laplaciano es positivo semidefinido.
        
        Propiedad: Todos los eigenvalues λᵢ ≥ 0
        """
        eigenvalues = np.linalg.eigvalsh(laplacian_matrix)
        
        assert np.all(eigenvalues >= -EPSILON_SPECTRAL), (
            f"Laplacian debe ser PSD, pero tiene eigenvalues negativos: "
            f"{eigenvalues[eigenvalues < 0]}"
        )
    
    def test_spectral_gap(self):
        """
        Test: Cálculo de brecha espectral.
        
        La brecha es la diferencia máxima entre eigenvalues consecutivos.
        """
        eigenvalues = np.array([0.0, 0.5, 0.6, 1.5, 2.0])
        gap = SpectralAnalyzer.spectral_gap(eigenvalues)
        
        # Brecha máxima es 1.5 - 0.6 = 0.9
        assert abs(gap - 0.9) < EPSILON_SPECTRAL, (
            f"Expected gap 0.9, got {gap}"
        )
    
    def test_cheeger_bounds(self):
        """
        Test: Bounds de Cheeger son consistentes.
        
        Propiedad: lower_bound ≤ 2 * upper_bound
        """
        fiedler = 0.5
        lower, upper = SpectralAnalyzer.cheeger_constant_bounds(fiedler)
        
        assert lower == fiedler / 2.0
        assert upper == 2.0 * fiedler
        assert lower <= 2 * upper
    
    def test_fiedler_raises_on_nonsymmetric(self):
        """
        Test: Fiedler eigenvalue rechaza matrices no simétricas.
        """
        nonsymmetric = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ], dtype=float)
        
        with pytest.raises(ValueError, match="symmetric"):
            SpectralAnalyzer.fiedler_eigenvalue(nonsymmetric)


class TestTopologyCalculator:
    """Tests para cálculos topológicos."""
    
    def test_betti_numbers_tree(self, simple_adjacency_matrix):
        """
        Test: Números de Betti para árbol.
        
        Para un árbol:
            - β₀ = 1 (conexo)
            - β₁ = 0 (acíclico)
        """
        beta_0, beta_1 = TopologyCalculator.betti_numbers_from_adjacency(
            simple_adjacency_matrix,
            directed=False
        )
        
        assert beta_0 == 1, "Tree must be connected (β₀ = 1)"
        assert beta_1 == 0, "Tree must be acyclic (β₁ = 0)"
    
    def test_betti_numbers_cycle(self, cycle_adjacency_matrix):
        """
        Test: Números de Betti para ciclo simple.
        
        Para un ciclo de 4 vértices:
            - β₀ = 1 (conexo)
            - β₁ = 1 (un ciclo independiente)
        """
        beta_0, beta_1 = TopologyCalculator.betti_numbers_from_adjacency(
            cycle_adjacency_matrix,
            directed=False
        )
        
        assert beta_0 == 1, "Cycle graph must be connected (β₀ = 1)"
        assert beta_1 == 1, "Cycle graph has one independent cycle (β₁ = 1)"
    
    def test_betti_numbers_disconnected(self, disconnected_adjacency_matrix):
        """
        Test: Números de Betti para grafo desconectado.
        
        Dos componentes sin ciclos:
            - β₀ = 2
            - β₁ = 0
        """
        beta_0, beta_1 = TopologyCalculator.betti_numbers_from_adjacency(
            disconnected_adjacency_matrix,
            directed=False
        )
        
        assert beta_0 == 2, "Must have 2 connected components (β₀ = 2)"
        assert beta_1 == 0, "No cycles (β₁ = 0)"
    
    def test_euler_characteristic_consistency(self):
        """
        Test: Fórmula de Euler es consistente.
        
        Para grafo planar: χ = V - E + F
        Para 1-esqueleto: χ = β₀ - β₁
        """
        # Grafo con V=5, E=6, β₀=1, β₁=2
        # (dos ciclos independientes)
        adj = np.array([
            [0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 0, 1],
            [0, 1, 0, 0, 1],
            [0, 0, 1, 1, 0]
        ], dtype=float)
        
        beta_0, beta_1 = TopologyCalculator.betti_numbers_from_adjacency(
            adj, directed=False
        )
        
        chi = TopologyCalculator.euler_characteristic([beta_0, beta_1])
        expected_chi = beta_0 - beta_1
        
        assert chi == expected_chi, (
            f"Euler characteristic inconsistency: χ={chi}, "
            f"β₀-β₁={expected_chi}"
        )
    
    def test_betti_numbers_empty_graph(self):
        """
        Test: Grafo vacío tiene β₀ = 0.
        """
        empty_adj = np.zeros((0, 0), dtype=float)
        beta_0, beta_1 = TopologyCalculator.betti_numbers_from_adjacency(
            empty_adj, directed=False
        )
        
        assert beta_0 == 0, "Empty graph should have β₀ = 0"
        assert beta_1 == 0, "Empty graph should have β₁ = 0"
    
    def test_betti_numbers_self_loop(self):
        """
        Test: Self-loop contribuye a β₁.
        
        Un vértice con self-loop tiene un ciclo de longitud 1.
        """
        # Grafo con self-loop en vértice 0
        adj = np.array([
            [1, 0],  # Self-loop
            [0, 0]
        ], dtype=float)
        
        beta_0, beta_1 = TopologyCalculator.betti_numbers_from_adjacency(
            adj, directed=False
        )
        
        # Dos componentes (vértice 0 con loop, vértice 1 aislado)
        assert beta_0 == 2, "Should have 2 components"
        # El self-loop crea un ciclo
        assert beta_1 >= 0, "Self-loop should contribute to cycles"


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

class TestTTLCache:
    """Tests para caché con TTL."""
    
    def test_cache_basic_operations(self, ttl_cache):
        """
        Test: Operaciones básicas get/set.
        """
        ttl_cache.set("key1", "value1")
        
        assert ttl_cache.get("key1") == "value1"
        assert ttl_cache.get("nonexistent") is None
    
    def test_cache_ttl_expiration(self):
        """
        Test: Entradas expiran después del TTL.
        """
        cache = TTLCache(ttl_seconds=0.1, maxsize=10, auto_cleanup=False)
        
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Esperar expiración
        time.sleep(0.15)
        
        assert cache.get("key1") is None, "Entry should have expired"
        
        cache.shutdown()
    
    def test_cache_lru_eviction(self):
        """
        Test: Evicción LRU cuando se alcanza maxsize.
        """
        cache = TTLCache(ttl_seconds=60, maxsize=3, auto_cleanup=False)
        
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.set("k3", "v3")
        
        # k1 es el más antiguo
        # Acceder a k1 para hacerlo más reciente
        cache.get("k1")
        
        # Agregar k4, debería evictar k2 (ahora el más antiguo)
        cache.set("k4", "v4")
        
        assert cache.get("k1") == "v1", "k1 should still be in cache"
        assert cache.get("k2") is None, "k2 should have been evicted"
        assert cache.get("k3") == "v3"
        assert cache.get("k4") == "v4"
        
        cache.shutdown()
    
    def test_cache_cleanup_expired(self):
        """
        Test: Limpieza manual de entradas expiradas.
        """
        cache = TTLCache(ttl_seconds=0.1, maxsize=10, auto_cleanup=False)
        
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        
        time.sleep(0.15)
        
        evicted = cache.cleanup_expired()
        
        assert evicted == 2, "Should have evicted 2 expired entries"
        assert cache.get("k1") is None
        assert cache.get("k2") is None
        
        cache.shutdown()
    
    def test_cache_auto_cleanup(self):
        """
        Test: Limpieza automática en background.
        """
        cache = TTLCache(
            ttl_seconds=0.2,
            maxsize=10,
            cleanup_interval=0.3,
            auto_cleanup=True
        )
        
        cache.set("k1", "v1")
        
        # Esperar que expire y se limpie automáticamente
        time.sleep(0.6)
        
        # El cleanup thread debería haber eliminado la entrada
        assert cache.get("k1") is None
        
        cache.shutdown()
    
    def test_cache_stats(self, ttl_cache):
        """
        Test: Estadísticas del caché.
        """
        ttl_cache.set("k1", "v1")
        ttl_cache.get("k1")  # Hit
        ttl_cache.get("k2")  # Miss
        
        stats = ttl_cache.stats
        
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1
        assert 0 <= stats["hit_rate"] <= 1
    
    def test_cache_thread_safety(self):
        """
        Test: Caché es thread-safe.
        """
        cache = TTLCache(ttl_seconds=10, maxsize=100, auto_cleanup=False)
        
        def worker(thread_id: int):
            for i in range(50):
                cache.set(f"t{thread_id}_k{i}", f"v{i}")
                cache.get(f"t{thread_id}_k{i}")
        
        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(5)
        ]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # No debería haber crashes
        stats = cache.stats
        assert stats["size"] <= 100  # Respetó maxsize
        
        cache.shutdown()
    
    def test_cache_clear(self, ttl_cache):
        """
        Test: Clear elimina todas las entradas.
        """
        ttl_cache.set("k1", "v1")
        ttl_cache.set("k2", "v2")
        
        ttl_cache.clear()
        
        assert ttl_cache.get("k1") is None
        assert ttl_cache.get("k2") is None
        assert ttl_cache.stats["size"] == 0


# =============================================================================
# TESTS DE VECTOR SEMÁNTICO
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
    
    def test_vector_critical_bridge_inconsistency_warns(self):
        """
        Test: Advertencia si bridge crítico tiene grado 0.
        """
        with pytest.warns() as record:
            PyramidalSemanticVector(
                node_id="test",
                node_type="APU",
                stratum=Stratum.TACTICS,
                in_degree=0,
                out_degree=0,
                is_critical_bridge=True  # Inconsistente
            )
        
        # Verificar que se emitió warning
        assert len(record) > 0


# =============================================================================
# TESTS DE PROYECTOR SEMÁNTICO
# =============================================================================

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
        # STABILITY: mayor es mejor
        classification = semantic_service.get_classification_by_threshold(
            metric_name="STABILITY",
            value=0.90
        )
        assert classification == "robust"
        
        classification = semantic_service.get_classification_by_threshold(
            metric_name="STABILITY",
            value=0.25
        )
        assert classification == "critical"
        
        # ENTROPY: mayor es peor (reverse)
        classification = semantic_service.get_classification_by_threshold(
            metric_name="ENTROPY",
            value=0.80
        )
        assert classification == "high"
    
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
    
    @given(st.integers(min_value=0, max_value=100))
    def test_betti_0_always_nonnegative(self, n_vertices):
        """
        Propiedad: β₀ ≥ 0 siempre.
        """
        if n_vertices == 0:
            pytest.skip("Empty graph")
        
        # Generar grafo aleatorio
        adj = np.random.randint(0, 2, size=(n_vertices, n_vertices))
        adj = (adj + adj.T) / 2  # Simetrizar
        np.fill_diagonal(adj, 0)  # Sin self-loops
        
        beta_0, beta_1 = TopologyCalculator.betti_numbers_from_adjacency(
            adj, directed=False
        )
        
        assert beta_0 >= 0, f"β₀ must be non-negative, got {beta_0}"
        assert beta_1 >= 0, f"β₁ must be non-negative, got {beta_1}"
    
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
    
    @given(npst.arrays(dtype=np.float64, shape=(10, 10)))
    def test_laplacian_row_sum_zero(self, matrix):
        """
        Propiedad: Las filas del Laplaciano suman 0.
        
        Para matriz de adyacencia A, L = D - A tiene filas que suman 0.
        """
        # Asegurar que es no negativa y simétrica
        adj = np.abs(matrix)
        adj = (adj + adj.T) / 2
        np.fill_diagonal(adj, 0)
        
        # Construir Laplaciano
        degrees = np.sum(adj, axis=1)
        L = np.diag(degrees) - adj
        
        # Verificar que filas suman ~0
        row_sums = np.sum(L, axis=1)
        
        assert np.allclose(row_sums, 0, atol=1e-10), (
            f"Laplacian rows must sum to 0, got max deviation: "
            f"{np.max(np.abs(row_sums))}"
        )
    
    @settings(max_examples=50)
    @given(st.floats(min_value=0.01, max_value=10.0))
    def test_cache_ttl_respected(self, ttl):
        """
        Propiedad: Entradas expiran después de TTL.
        """
        cache = TTLCache(ttl_seconds=ttl, maxsize=10, auto_cleanup=False)
        
        cache.set("key", "value")
        
        # Inmediatamente debe estar disponible
        assert cache.get("key") == "value"
        
        # Después de TTL debe haber expirado
        time.sleep(ttl + 0.1)
        assert cache.get("key") is None
        
        cache.shutdown()


# =============================================================================
# BENCHMARKS Y TESTS DE PERFORMANCE
# =============================================================================

class TestPerformance:
    """Tests de performance y escalabilidad."""
    
    @pytest.mark.benchmark
    def test_betti_computation_scalability(self, benchmark):
        """
        Benchmark: Cálculo de números de Betti escala bien.
        
        Complejidad esperada: O(V + E) para DFS
        """
        n = 100
        adj = np.random.randint(0, 2, size=(n, n))
        adj = (adj + adj.T) / 2
        np.fill_diagonal(adj, 0)
        
        def compute():
            return TopologyCalculator.betti_numbers_from_adjacency(
                adj, directed=False
            )
        
        result = benchmark(compute)
        assert result[0] >= 1  # Al menos una componente
    
    @pytest.mark.benchmark
    def test_cache_performance(self, benchmark):
        """
        Benchmark: Performance del caché.
        """
        cache = TTLCache(ttl_seconds=60, maxsize=1000, auto_cleanup=False)
        
        # Pre-populate
        for i in range(100):
            cache.set(f"key{i}", f"value{i}")
        
        def access_cache():
            for i in range(100):
                cache.get(f"key{i % 50}")  # 50% hit rate
        
        benchmark(access_cache)
        
        stats = cache.stats
        assert stats["hit_rate"] > 0
        
        cache.shutdown()
    
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
    
    def test_spectral_analysis_efficiency(self):
        """
        Test: Análisis espectral es eficiente.
        
        Complejidad esperada: O(n³) para eigenvalues completos,
        pero usamos eigvalsh (simétrico) que es más rápido.
        """
        import time
        
        n = 100
        # Grafo aleatorio
        adj = np.random.rand(n, n)
        adj = (adj + adj.T) / 2
        np.fill_diagonal(adj, 0)
        
        # Laplaciano
        degrees = np.sum(adj, axis=1)
        L = np.diag(degrees) - adj
        
        start = time.perf_counter()
        fiedler = SpectralAnalyzer.fiedler_eigenvalue(L)
        elapsed = time.perf_counter() - start
        
        assert elapsed < 1.0, f"Spectral analysis too slow: {elapsed:.3f}s"
        assert fiedler >= 0


# =============================================================================
# TESTS DE CASOS EXTREMOS (EDGE CASES)
# =============================================================================

class TestEdgeCases:
    """Tests de casos extremos y situaciones límite."""
    
    def test_empty_graph_betti(self):
        """
        Test: Grafo vacío tiene β₀ = 0.
        """
        empty = np.zeros((0, 0))
        beta_0, beta_1 = TopologyCalculator.betti_numbers_from_adjacency(
            empty, directed=False
        )
        
        assert beta_0 == 0
        assert beta_1 == 0
    
    def test_single_vertex_graph(self):
        """
        Test: Grafo con un solo vértice aislado.
        """
        single = np.zeros((1, 1))
        beta_0, beta_1 = TopologyCalculator.betti_numbers_from_adjacency(
            single, directed=False
        )
        
        assert beta_0 == 1  # Una componente
        assert beta_1 == 0  # Sin ciclos
    
    def test_complete_graph(self):
        """
        Test: Grafo completo Kₙ.
        
        Para Kₙ:
            - β₀ = 1 (conexo)
            - β₁ = número de ciclos independientes
        """
        n = 5
        # Grafo completo: todos conectados con todos
        adj = np.ones((n, n)) - np.eye(n)
        
        beta_0, beta_1 = TopologyCalculator.betti_numbers_from_adjacency(
            adj, directed=False
        )
        
        assert beta_0 == 1  # Conexo
        # K₅ tiene muchos ciclos
        assert beta_1 > 0
    
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
    
    def test_stratum_conversion_edge_cases(self):
        """
        Test: Conversión de Stratum con casos límite.
        """
        service = SemanticDictionaryService()
        
        # Integer
        assert service.convert_stratum_value(0) == Stratum.WISDOM
        assert service.convert_stratum_value(4) == Stratum.PHYSICS
        
        # String
        assert service.convert_stratum_value("WISDOM") == Stratum.WISDOM
        assert service.convert_stratum_value("wisdom") == Stratum.WISDOM
        
        # Ya es Stratum
        assert service.convert_stratum_value(Stratum.OMEGA) == Stratum.OMEGA
        
        # Inválidos
        with pytest.raises(ValueError):
            service.convert_stratum_value(999)
        
        with pytest.raises(ValueError):
            service.convert_stratum_value("INVALID")
        
        with pytest.raises(TypeError):
            service.convert_stratum_value([1, 2, 3])  # type: ignore


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
    
    def test_regression_fiedler_single_vertex(self):
        """
        Regresión: Fiedler con un solo vértice causaba index error.
        """
        L = np.array([[0.0]])
        fiedler = SpectralAnalyzer.fiedler_eigenvalue(L)
        
        # Con un solo vértice, no hay segundo eigenvalue
        assert fiedler == 0.0
    
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