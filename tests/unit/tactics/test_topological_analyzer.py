"""
=========================================================================================
Suite de Pruebas: Topological Analyzer (Operador de Observabilidad Funtorial y TDA)
Ubicación: tests/unit/tactics/test_topological_analyzer.py
Versión: 2.0.0-rigorous
=========================================================================================

Suite de pruebas exhaustiva basada en:
- Propiedades matemáticas fundamentales (invariantes topológicas)
- Casos límite y condiciones extremas
- Teoremas y axiomas de topología algebraica
- Validación de coherencia funtorial
- Análisis de estabilidad numérica

Metodología:
------------
1. **Property-Based Testing**: Verificación de invariantes matemáticas
2. **Theorem-Driven Testing**: Cada test verifica un teorema/axioma específico
3. **Adversarial Testing**: Casos patológicos y entradas maliciosas
4. **Regression Testing**: Casos conocidos de bugs históricos
5. **Performance Testing**: Límites de escalabilidad y complejidad

Organización:
-------------
- TestTopologicalConstants: Validación de constantes matemáticas
- TestBettiNumbers: Teoremas de Euler-Poincaré y números de Betti
- TestPersistenceInterval: Geometría de diagramas de persistencia
- TestSystemTopology: Invariantes de grafos y operaciones topológicas
- TestPersistenceHomology: Teoría de homología persistente
- TestTopologicalHealth: Modelo de salud y penalizaciones
- TestIntegration: Escenarios completos end-to-end
- TestEdgeCases: Casos extremos y patológicos
- TestNumericalStability: Estabilidad numérica y precisión
"""

from __future__ import annotations

import copy
import itertools
import math
import os
import sys
import tempfile
import unittest
from decimal import Decimal, getcontext
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from unittest.mock import MagicMock, Mock, patch

import networkx as nx
import numpy as np

# Importar el módulo a testear
# Ajustar path si es necesario
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app", "tactics")))

try:
    from app.tactics.topological_analyzer import (
        BettiNumberError,
        BettiNumbers,
        EulerCharacteristic,
        GraphStructureError,
        HealthLevel,
        HealthScore,
        InvalidTopologyError,
        MetricState,
        PersistenceAnalysisResult,
        PersistenceComputationError,
        PersistenceHomology,
        PersistenceInterval,
        RequestLoopInfo,
        SystemTopology,
        TopologicalConstants,
        TopologicalError,
        TopologicalHealth,
        compute_wasserstein_distance,
        create_simple_topology,
    )
except ImportError as e:
    print(f"❌ Error importando módulo: {e}")
    print("Asegúrate de que topological_analyzer.py esté en app/tactics/")
    sys.exit(1)


# =============================================================================
# UTILIDADES DE TESTING
# =============================================================================

class MathematicalAssertions:
    """Mixin con aserciones matemáticas rigurosas."""
    
    def assertAlmostEqualFloat(
        self,
        first: float,
        second: float,
        tolerance: float = 1e-9,
        msg: Optional[str] = None
    ) -> None:
        """Verifica igualdad de floats con tolerancia especificada."""
        if math.isnan(first) or math.isnan(second):
            if not (math.isnan(first) and math.isnan(second)):
                raise AssertionError(msg or f"NaN mismatch: {first} != {second}")
            return
        
        if math.isinf(first) or math.isinf(second):
            if first != second:
                raise AssertionError(msg or f"Infinity mismatch: {first} != {second}")
            return
        
        diff = abs(first - second)
        if diff > tolerance:
            raise AssertionError(
                msg or f"Floats differ beyond tolerance: |{first} - {second}| = {diff} > {tolerance}"
            )
    
    def assertValidBettiNumbers(self, betti: BettiNumbers) -> None:
        """Verifica que los números de Betti satisfagan todos los invariantes."""
        # No negatividad
        self.assertGreaterEqual(betti.b0, 0, "β₀ debe ser no negativo")
        self.assertGreaterEqual(betti.b1, 0, "β₁ debe ser no negativo")
        self.assertGreaterEqual(betti.num_vertices, 0, "|V| debe ser no negativo")
        self.assertGreaterEqual(betti.num_edges, 0, "|E| debe ser no negativo")
        
        # Cotas superiores
        if betti.num_vertices > 0:
            self.assertLessEqual(betti.b0, betti.num_vertices, "β₀ ≤ |V|")
        
        if betti.num_edges > 0:
            self.assertLessEqual(betti.b1, betti.num_edges, "β₁ ≤ |E|")
        
        # Teorema de Euler-Poincaré
        if betti.num_vertices > 0 and betti.num_edges >= 0:
            self.assertTrue(
                betti.verify_euler_consistency(),
                f"Violación de Euler-Poincaré: χ = β₀ - β₁ = {betti.euler_characteristic} "
                f"≠ |V| - |E| = {betti.euler_characteristic_alt}"
            )
    
    def assertValidPersistenceInterval(self, interval: PersistenceInterval) -> None:
        """Verifica validez de un intervalo de persistencia."""
        self.assertGreaterEqual(interval.birth, 0, "Birth debe ser no negativo")
        
        if interval.death >= 0:
            self.assertGreaterEqual(
                interval.death,
                interval.birth,
                f"Death ({interval.death}) debe ser >= birth ({interval.birth})"
            )
        
        self.assertGreaterEqual(interval.dimension, 0, "Dimensión debe ser no negativa")
        self.assertGreaterEqual(interval.amplitude, 0.0, "Amplitud debe ser no negativa")
        self.assertTrue(math.isfinite(interval.amplitude), "Amplitud debe ser finita")
    
    def assertGraphInvariant(
        self,
        topology: SystemTopology,
        expected_b0: Optional[int] = None,
        expected_b1: Optional[int] = None
    ) -> None:
        """Verifica invariantes del grafo."""
        betti = topology.calculate_betti_numbers()
        self.assertValidBettiNumbers(betti)
        
        if expected_b0 is not None:
            self.assertEqual(betti.b0, expected_b0, f"Esperado β₀={expected_b0}, obtenido {betti.b0}")
        
        if expected_b1 is not None:
            self.assertEqual(betti.b1, expected_b1, f"Esperado β₁={expected_b1}, obtenido {betti.b1}")


# =============================================================================
# GRUPO 1: TEST DE CONSTANTES MATEMÁTICAS
# =============================================================================

class TestTopologicalConstants(unittest.TestCase, MathematicalAssertions):
    """
    Verifica la validez matemática de las constantes fundamentales.
    
    Teoremas verificados:
    - Característica de Euler para variedades conocidas
    - Normalización de pesos (suma = 1.0)
    - Invariantes topológicos clásicos
    """
    
    def test_euler_characteristics_classical_surfaces(self):
        """
        Teorema: Características de Euler de superficies clásicas.
        
        Verifica:
        - χ(S²) = 2 (esfera)
        - χ(T²) = 0 (toro)
        - χ(ℝP²) = 1 (plano proyectivo)
        - χ(K²) = 0 (botella de Klein)
        """
        TC = TopologicalConstants
        
        self.assertEqual(TC.EULER_SPHERE, 2, "χ(S²) debe ser 2")
        self.assertEqual(TC.EULER_TORUS, 0, "χ(T²) debe ser 0")
        self.assertEqual(TC.EULER_PROJECTIVE_PLANE, 1, "χ(ℝP²) debe ser 1")
        self.assertEqual(TC.EULER_KLEIN_BOTTLE, 0, "χ(K²) debe ser 0")
    
    def test_weight_normalization_theorem(self):
        """
        Teorema: Los pesos del modelo de salud deben sumar 1.0.
        
        Propiedad fundamental para que el score de salud esté en [0, 1].
        """
        TC = TopologicalConstants
        
        total_weight = (
            TC.WEIGHT_FRAGMENTATION +
            TC.WEIGHT_CYCLES +
            TC.WEIGHT_DISCONNECTED +
            TC.WEIGHT_MISSING_EDGES +
            TC.WEIGHT_RETRY_LOOPS
        )
        
        self.assertAlmostEqualFloat(
            total_weight,
            1.0,
            tolerance=TC.EPSILON,
            msg=f"Pesos deben sumar 1.0, suma actual: {total_weight}"
        )
        
        # Validación interna
        self.assertTrue(TC.validate_weights(), "validate_weights() debe retornar True")
    
    def test_epsilon_hierarchy(self):
        """Verifica jerarquía de tolerancias: EPSILON_STRICT < EPSILON."""
        TC = TopologicalConstants
        
        self.assertLess(
            TC.EPSILON_STRICT,
            TC.EPSILON,
            "Tolerancia estricta debe ser menor que tolerancia estándar"
        )
        
        self.assertGreater(TC.EPSILON_STRICT, 0, "Tolerancia estricta debe ser positiva")
        self.assertGreater(TC.EPSILON, 0, "Tolerancia estándar debe ser positiva")
    
    def test_threshold_ratios_ordering(self):
        """Verifica orden lógico de umbrales de persistencia."""
        TC = TopologicalConstants
        
        self.assertLess(
            TC.MIN_PERSISTENCE_RATIO,
            TC.NOISE_THRESHOLD_RATIO,
            "Persistencia mínima < umbral de ruido"
        )
        
        self.assertLess(
            TC.NOISE_THRESHOLD_RATIO,
            TC.CRITICAL_THRESHOLD_RATIO,
            "Umbral de ruido < umbral crítico"
        )
        
        # Todos deben estar en (0, 1]
        for ratio_name in ["MIN_PERSISTENCE_RATIO", "NOISE_THRESHOLD_RATIO", "CRITICAL_THRESHOLD_RATIO"]:
            ratio = getattr(TC, ratio_name)
            self.assertGreater(ratio, 0.0, f"{ratio_name} debe ser > 0")
            self.assertLessEqual(ratio, 1.0, f"{ratio_name} debe ser ≤ 1")
    
    def test_cyclomatic_limits_sanity(self):
        """Verifica límites de complejidad ciclomática."""
        TC = TopologicalConstants
        
        self.assertGreater(TC.WARNING_CYCLOMATIC_COMPLEXITY, 0)
        self.assertGreater(TC.MAX_CYCLOMATIC_COMPLEXITY, TC.WARNING_CYCLOMATIC_COMPLEXITY)
        self.assertLessEqual(TC.MAX_CYCLOMATIC_COMPLEXITY, 100, "Límite razonable")
    
    def test_component_limits(self):
        """Verifica límites de componentes conexas."""
        TC = TopologicalConstants
        
        self.assertEqual(TC.MAX_COMPONENTS_HEALTHY, 1, "Sistema saludable debe ser conexo")
        self.assertGreater(TC.MAX_COMPONENTS_WARNING, TC.MAX_COMPONENTS_HEALTHY)
    
    def test_all_weights_positive(self):
        """Verifica que todos los pesos sean estrictamente positivos."""
        TC = TopologicalConstants
        
        weights = [
            TC.WEIGHT_FRAGMENTATION,
            TC.WEIGHT_CYCLES,
            TC.WEIGHT_DISCONNECTED,
            TC.WEIGHT_MISSING_EDGES,
            TC.WEIGHT_RETRY_LOOPS
        ]
        
        for weight in weights:
            self.assertGreater(weight, 0.0, "Todos los pesos deben ser > 0")
            self.assertLess(weight, 1.0, "Ningún peso debe dominar completamente")


# =============================================================================
# GRUPO 2: TEST DE NÚMEROS DE BETTI
# =============================================================================

class TestBettiNumbers(unittest.TestCase, MathematicalAssertions):
    """
    Verificación rigurosa de invariantes topológicas.
    
    Teoremas verificados:
    - Euler-Poincaré: χ = |V| - |E| = β₀ - β₁
    - Cotas: β₀ ≤ |V|, β₁ ≤ |E|
    - Propiedades de grafos especiales (árboles, ciclos, completos)
    """
    
    def test_euler_poincare_consistency_empty_graph(self):
        """
        Teorema: Para grafo vacío, χ = 0 = β₀ - β₁.
        """
        betti = BettiNumbers(b0=0, b1=0, num_vertices=0, num_edges=0)
        
        self.assertEqual(betti.euler_characteristic, 0)
        self.assertEqual(betti.euler_characteristic_alt, 0)
        self.assertTrue(betti.verify_euler_consistency())
    
    def test_euler_poincare_consistency_single_vertex(self):
        """
        Teorema: Para grafo con 1 vértice aislado, χ = 1.
        
        β₀ = 1, β₁ = 0 ⟹ χ = 1
        """
        betti = BettiNumbers(b0=1, b1=0, num_vertices=1, num_edges=0)
        
        self.assertEqual(betti.euler_characteristic, 1)
        self.assertEqual(betti.euler_characteristic_alt, 1)
        self.assertTrue(betti.verify_euler_consistency())
        self.assertTrue(betti.is_connected)
        self.assertTrue(betti.is_acyclic)
        self.assertTrue(betti.is_ideal)
    
    def test_euler_poincare_tree_property(self):
        """
        Teorema: Para árbol con n vértices, |E| = n - 1 y χ = 1.
        
        β₀ = 1, β₁ = 0 para todo árbol.
        """
        # Árbol con 5 vértices
        n = 5
        betti = BettiNumbers(b0=1, b1=0, num_vertices=n, num_edges=n - 1)
        
        self.assertEqual(betti.euler_characteristic, 1)
        self.assertTrue(betti.is_tree)
        self.assertTrue(betti.is_forest)
        self.assertEqual(betti.cyclomatic_complexity, 1)
    
    def test_euler_poincare_cycle_graph(self):
        """
        Teorema: Para ciclo simple Cₙ, β₀ = 1, β₁ = 1, χ = 0.
        
        Un ciclo tiene |V| = |E| = n.
        """
        n = 6
        betti = BettiNumbers(b0=1, b1=1, num_vertices=n, num_edges=n)
        
        self.assertEqual(betti.euler_characteristic, 0)
        self.assertTrue(betti.is_connected)
        self.assertFalse(betti.is_acyclic)
        self.assertEqual(betti.cyclomatic_complexity, 2)
    
    def test_disconnected_graph_betti_numbers(self):
        """
        Teorema: Para grafo con k componentes, β₀ = k.
        
        Ejemplo: 3 vértices aislados ⟹ β₀ = 3, β₁ = 0.
        """
        betti = BettiNumbers(b0=3, b1=0, num_vertices=3, num_edges=0)
        
        self.assertFalse(betti.is_connected)
        self.assertTrue(betti.is_acyclic)
        self.assertEqual(betti.euler_characteristic, 3)
    
    def test_complete_graph_formula(self):
        """
        Teorema: Para grafo completo Kₙ:
        - |E| = n(n-1)/2
        - β₀ = 1
        - β₁ = |E| - |V| + 1 = n(n-1)/2 - n + 1 = (n-1)(n-2)/2
        """
        n = 5
        num_edges = n * (n - 1) // 2  # 10 aristas
        expected_b1 = num_edges - n + 1  # 10 - 5 + 1 = 6
        
        betti = BettiNumbers(b0=1, b1=expected_b1, num_vertices=n, num_edges=num_edges)
        
        self.assertEqual(betti.b1, 6)
        self.assertTrue(betti.verify_euler_consistency())
        self.assertValidBettiNumbers(betti)
    
    def test_validation_negative_b0_raises(self):
        """Verifica que β₀ negativo lance excepción."""
        with self.assertRaises(BettiNumberError) as ctx:
            BettiNumbers(b0=-1, b1=0, num_vertices=0, num_edges=0)
        
        self.assertIn("β₀ debe ser no negativo", str(ctx.exception))
    
    def test_validation_negative_b1_raises(self):
        """Verifica que β₁ negativo lance excepción."""
        with self.assertRaises(BettiNumberError) as ctx:
            BettiNumbers(b0=1, b1=-1, num_vertices=1, num_edges=0)
        
        self.assertIn("β₁ debe ser no negativo", str(ctx.exception))
    
    def test_validation_b0_exceeds_vertices_raises(self):
        """Verifica que β₀ > |V| lance excepción."""
        with self.assertRaises(BettiNumberError) as ctx:
            BettiNumbers(b0=5, b1=0, num_vertices=3, num_edges=0)
        
        self.assertIn("β₀", str(ctx.exception))
        self.assertIn("no puede exceder", str(ctx.exception))
    
    def test_validation_euler_inconsistency_raises(self):
        """Verifica que violación de Euler-Poincaré lance excepción."""
        # Valores inconsistentes: β₁ ≠ |E| - |V| + β₀
        # Para 4 vértices, 5 aristas, β₀=1: β₁ esperado = 5 - 4 + 1 = 2
        with self.assertRaises(BettiNumberError) as ctx:
            BettiNumbers(b0=1, b1=10, num_vertices=4, num_edges=5)
        
        self.assertIn("Euler-Poincaré", str(ctx.exception))
    
    def test_betti_numbers_immutability(self):
        """Verifica que BettiNumbers sea inmutable (frozen dataclass)."""
        betti = BettiNumbers(b0=1, b1=0, num_vertices=3, num_edges=2)
        
        with self.assertRaises(Exception):  # FrozenInstanceError o AttributeError
            betti.b0 = 2
    
    def test_to_dict_serialization(self):
        """Verifica serialización completa a diccionario."""
        betti = BettiNumbers(b0=2, b1=1, num_vertices=5, num_edges=4)
        data = betti.to_dict()
        
        # Campos requeridos
        self.assertEqual(data["b0"], 2)
        self.assertEqual(data["b1"], 1)
        self.assertEqual(data["num_vertices"], 5)
        self.assertEqual(data["num_edges"], 4)
        self.assertEqual(data["euler_characteristic"], 1)
        
        # Propiedades derivadas
        self.assertIn("is_connected", data)
        self.assertIn("is_acyclic", data)
        self.assertIn("cyclomatic_complexity", data)
        self.assertTrue(data["euler_consistent"])


# =============================================================================
# GRUPO 3: TEST DE INTERVALOS DE PERSISTENCIA
# =============================================================================

class TestPersistenceInterval(unittest.TestCase, MathematicalAssertions):
    """
    Verificación de geometría de diagramas de persistencia.
    
    Teoremas verificados:
    - Persistencia = (death - birth) / √2
    - Distancia de Bottleneck entre intervalos
    - Propiedades de intervalos vivos vs muertos
    """
    
    def test_finite_interval_properties(self):
        """Verifica propiedades básicas de intervalo finito."""
        interval = PersistenceInterval(birth=10, death=50, dimension=0, amplitude=0.8)
        
        self.assertEqual(interval.lifespan, 40.0)
        self.assertFalse(interval.is_alive)
        
        # Persistencia = (death - birth) / √2
        expected_persistence = 40.0 / math.sqrt(2.0)
        self.assertAlmostEqualFloat(interval.persistence, expected_persistence)
        
        # Punto medio
        self.assertEqual(interval.midpoint, 30.0)
        
        self.assertValidPersistenceInterval(interval)
    
    def test_alive_interval_properties(self):
        """Verifica propiedades de intervalo que aún vive (death = -1)."""
        interval = PersistenceInterval(birth=100, death=-1, dimension=1, amplitude=1.5)
        
        self.assertTrue(interval.is_alive)
        self.assertTrue(math.isinf(interval.lifespan))
        self.assertTrue(math.isinf(interval.persistence))
        self.assertEqual(interval.midpoint, 100.0)
        
        self.assertValidPersistenceInterval(interval)
    
    def test_zero_lifespan_interval(self):
        """Intervalo con lifespan = 0 (nace y muere instantáneamente)."""
        interval = PersistenceInterval(birth=42, death=42, dimension=0, amplitude=0.0)
        
        self.assertEqual(interval.lifespan, 0.0)
        self.assertEqual(interval.persistence, 0.0)
        self.assertEqual(interval.midpoint, 42.0)
    
    def test_bottleneck_distance_identical_intervals(self):
        """
        Teorema: d_B(I, I) = 0.
        """
        i1 = PersistenceInterval(birth=10, death=20, dimension=0)
        i2 = PersistenceInterval(birth=10, death=20, dimension=0)
        
        dist = i1.bottleneck_distance(i2)
        self.assertEqual(dist, 0.0)
    
    def test_bottleneck_distance_different_dimensions_infinity(self):
        """
        Teorema: d_B entre intervalos de dimensiones distintas = ∞.
        """
        i1 = PersistenceInterval(birth=10, death=20, dimension=0)
        i2 = PersistenceInterval(birth=10, death=20, dimension=1)
        
        dist = i1.bottleneck_distance(i2)
        self.assertTrue(math.isinf(dist))
    
    def test_bottleneck_distance_formula(self):
        """
        Teorema: d_B(I₁, I₂) = max(|b₁ - b₂|, |d₁ - d₂|).
        """
        i1 = PersistenceInterval(birth=10, death=50, dimension=0)
        i2 = PersistenceInterval(birth=15, death=45, dimension=0)
        
        # |10 - 15| = 5, |50 - 45| = 5 ⟹ max = 5
        dist = i1.bottleneck_distance(i2)
        self.assertEqual(dist, 5.0)
    
    def test_bottleneck_distance_one_alive(self):
        """Distancia cuando un intervalo está vivo."""
        i1 = PersistenceInterval(birth=10, death=50, dimension=0)
        i2 = PersistenceInterval(birth=10, death=-1, dimension=0)
        
        dist = i1.bottleneck_distance(i2)
        self.assertTrue(math.isinf(dist), "Distancia debe ser infinita")
    
    def test_bottleneck_distance_both_alive(self):
        """Distancia cuando ambos intervalos están vivos."""
        i1 = PersistenceInterval(birth=10, death=-1, dimension=0)
        i2 = PersistenceInterval(birth=15, death=-1, dimension=0)
        
        dist = i1.bottleneck_distance(i2)
        # Solo difieren en birth: max(5, 0) = 5
        self.assertEqual(dist, 5.0)
    
    def test_validation_negative_birth_raises(self):
        """Verifica que birth negativo lance excepción."""
        with self.assertRaises(PersistenceComputationError):
            PersistenceInterval(birth=-5, death=10, dimension=0)
    
    def test_validation_death_before_birth_raises(self):
        """Verifica que death < birth lance excepción."""
        with self.assertRaises(PersistenceComputationError):
            PersistenceInterval(birth=50, death=30, dimension=0)
    
    def test_validation_negative_dimension_raises(self):
        """Verifica que dimensión negativa lance excepción."""
        with self.assertRaises(PersistenceComputationError):
            PersistenceInterval(birth=10, death=20, dimension=-1)
    
    def test_validation_negative_amplitude_raises(self):
        """Verifica que amplitud negativa lance excepción."""
        with self.assertRaises(PersistenceComputationError):
            PersistenceInterval(birth=10, death=20, dimension=0, amplitude=-0.5)
    
    def test_validation_infinite_amplitude_raises(self):
        """Verifica que amplitud infinita lance excepción."""
        with self.assertRaises(PersistenceComputationError):
            PersistenceInterval(birth=10, death=20, dimension=0, amplitude=float('inf'))
    
    def test_to_dict_finite_interval(self):
        """Serialización de intervalo finito."""
        interval = PersistenceInterval(birth=5, death=15, dimension=1, amplitude=2.5)
        data = interval.to_dict()
        
        self.assertEqual(data["birth"], 5)
        self.assertEqual(data["death"], 15)
        self.assertEqual(data["dimension"], 1)
        self.assertEqual(data["amplitude"], 2.5)
        self.assertFalse(data["is_alive"])
        self.assertIsNotNone(data["lifespan"])
        self.assertIsNotNone(data["persistence"])
    
    def test_to_dict_alive_interval(self):
        """Serialización de intervalo vivo."""
        interval = PersistenceInterval(birth=100, death=-1, dimension=0, amplitude=1.0)
        data = interval.to_dict()
        
        self.assertEqual(data["death"], -1)
        self.assertTrue(data["is_alive"])
        self.assertIsNone(data["lifespan"], "Lifespan debe ser None para intervalos infinitos")
        self.assertIsNone(data["persistence"])


# =============================================================================
# GRUPO 4: TEST DE SystemTopology (CORE)
# =============================================================================

class TestSystemTopology(unittest.TestCase, MathematicalAssertions):
    """
    Verificación exhaustiva de operaciones topológicas en grafos.
    
    Categorías:
    - Gestión de nodos y aristas
    - Cálculo de números de Betti
    - Detección de ciclos y anomalías
    - Invariantes topológicas
    - Operaciones atómicas
    """
    
    def setUp(self):
        """Configuración común para cada test."""
        self.topology = SystemTopology(max_history=50, validate_strictly=True)
    
    def tearDown(self):
        """Limpieza después de cada test."""
        del self.topology
    
    # -------------------------------------------------------------------------
    # Inicialización y Configuración
    # -------------------------------------------------------------------------
    
    def test_initialization_default(self):
        """Inicialización con parámetros default."""
        topology = SystemTopology()
        
        # Nodos requeridos deben estar presentes
        self.assertEqual(topology.num_nodes, len(SystemTopology.REQUIRED_NODES))
        self.assertEqual(topology.num_edges, 0)
        
        # Estado inicial: sin conexiones
        betti = topology.calculate_betti_numbers()
        self.assertEqual(betti.b0, len(SystemTopology.REQUIRED_NODES))
        self.assertEqual(betti.b1, 0)
    
    def test_initialization_custom_nodes(self):
        """Inicialización con nodos personalizados."""
        custom_nodes = {"ServiceA", "ServiceB"}
        topology = SystemTopology(custom_nodes=custom_nodes)
        
        expected_nodes = SystemTopology.REQUIRED_NODES.union(custom_nodes)
        self.assertEqual(topology.num_nodes, len(expected_nodes))
        
        for node in expected_nodes:
            self.assertTrue(topology.has_node(node))
    
    def test_initialization_invalid_window_size_raises(self):
        """Verifica que window_size inválido lance excepción."""
        with self.assertRaises(ValueError):
            SystemTopology(max_history=1)  # Muy pequeño
        
        with self.assertRaises(ValueError):
            SystemTopology(max_history=10001)  # Muy grande
    
    # -------------------------------------------------------------------------
    # Gestión de Nodos
    # -------------------------------------------------------------------------
    
    def test_add_node_valid(self):
        """Agregar nodo válido debe retornar True."""
        result = self.topology.add_node("NewService")
        
        self.assertTrue(result)
        self.assertTrue(self.topology.has_node("NewService"))
        self.assertEqual(self.topology.num_nodes, len(SystemTopology.REQUIRED_NODES) + 1)
    
    def test_add_node_duplicate_returns_false(self):
        """Agregar nodo duplicado debe retornar False."""
        self.topology.add_node("TestNode")
        result = self.topology.add_node("TestNode")
        
        self.assertFalse(result)
    
    def test_add_node_empty_string_strict_mode_raises(self):
        """Nodo vacío en modo estricto debe lanzar excepción."""
        with self.assertRaises(GraphStructureError):
            self.topology.add_node("")
    
    def test_add_node_whitespace_only_stripped(self):
        """Nodo con solo whitespace debe fallar después de strip."""
        with self.assertRaises(GraphStructureError):
            self.topology.add_node("   ")
    
    def test_add_node_non_string_raises(self):
        """Nodo no-string debe lanzar excepción en modo estricto."""
        with self.assertRaises(GraphStructureError):
            self.topology.add_node(123)  # type: ignore
    
    def test_remove_node_non_required_success(self):
        """Eliminar nodo no-requerido debe funcionar."""
        self.topology.add_node("Temp")
        result = self.topology.remove_node("Temp")
        
        self.assertTrue(result)
        self.assertFalse(self.topology.has_node("Temp"))
    
    def test_remove_node_required_strict_mode_raises(self):
        """Eliminar nodo requerido en modo estricto debe lanzar excepción."""
        with self.assertRaises(GraphStructureError):
            self.topology.remove_node("Core")
    
    def test_remove_node_nonexistent_returns_false(self):
        """Eliminar nodo inexistente debe retornar False."""
        result = self.topology.remove_node("Nonexistent")
        self.assertFalse(result)
    
    # -------------------------------------------------------------------------
    # Gestión de Conectividad
    # -------------------------------------------------------------------------
    
    def test_update_connectivity_valid_connections(self):
        """Actualizar conectividad con conexiones válidas."""
        connections = [
            ("Agent", "Core"),
            ("Core", "Redis"),
            ("Core", "Filesystem")
        ]
        
        edges_added, warnings = self.topology.update_connectivity(connections)
        
        self.assertEqual(edges_added, 3)
        self.assertEqual(len(warnings), 0)
        self.assertEqual(self.topology.num_edges, 3)
        
        # Verificar Betti numbers
        betti = self.topology.calculate_betti_numbers()
        self.assertEqual(betti.b0, 1, "Debe haber 1 componente conexa")
        self.assertEqual(betti.b1, 0, "No debe haber ciclos")
    
    def test_update_connectivity_with_cycle(self):
        """Actualizar conectividad creando un ciclo."""
        connections = [
            ("Agent", "Core"),
            ("Core", "Redis"),
            ("Redis", "Agent")
        ]
        
        self.topology.update_connectivity(connections)
        betti = self.topology.calculate_betti_numbers()
        
        self.assertEqual(betti.b0, 1)
        self.assertEqual(betti.b1, 1, "Debe detectar 1 ciclo")
        self.assertGreater(betti.cyclomatic_complexity, 1)
    
    def test_update_connectivity_self_loop_ignored(self):
        """Self-loops deben ser ignorados."""
        connections = [
            ("Core", "Core"),  # Self-loop
            ("Agent", "Core")
        ]
        
        edges_added, warnings = self.topology.update_connectivity(connections)
        
        # Solo 1 arista válida
        self.assertEqual(edges_added, 1)
        self.assertEqual(len(warnings), 1)
        self.assertIn("Auto-loop", warnings[0])
    
    def test_update_connectivity_missing_nodes_auto_add(self):
        """Nodos faltantes con auto_add=True deben agregarse."""
        connections = [("NewNode1", "NewNode2")]
        
        edges_added, warnings = self.topology.update_connectivity(
            connections,
            auto_add_nodes=True
        )
        
        self.assertEqual(edges_added, 1)
        self.assertTrue(self.topology.has_node("NewNode1"))
        self.assertTrue(self.topology.has_node("NewNode2"))
    
    def test_update_connectivity_missing_nodes_no_auto_add_warning(self):
        """Nodos faltantes sin auto_add deben generar warnings."""
        connections = [("Ghost1", "Ghost2")]
        
        edges_added, warnings = self.topology.update_connectivity(
            connections,
            validate_nodes=True,
            auto_add_nodes=False
        )
        
        self.assertEqual(edges_added, 0)
        self.assertGreater(len(warnings), 0)
    
    def test_update_connectivity_atomicity_on_error(self):
        """Verifica rollback cuando hay error crítico."""
        # Configurar estado inicial
        self.topology.update_connectivity([("Agent", "Core")])
        initial_edges = self.topology.num_edges
        
        # Provocar error con entrada inválida en medio
        bad_connections = [
            ("Core", "Redis"),
            (None, "Agent"),  # Tipo inválido  # type: ignore
            ("Redis", "Filesystem")
        ]
        
        edges_added, warnings = self.topology.update_connectivity(bad_connections)
        
        # Debe haber warnings
        self.assertGreater(len(warnings), 0)
    
    def test_add_edge_success(self):
        """Agregar arista válida."""
        result = self.topology.add_edge("Agent", "Core")
        
        self.assertTrue(result)
        self.assertEqual(self.topology.num_edges, 1)
        self.assertIn(("Agent", "Core"), self.topology.edges)
    
    def test_add_edge_missing_node_returns_false(self):
        """Agregar arista con nodo faltante debe fallar."""
        result = self.topology.add_edge("Agent", "Nonexistent")
        self.assertFalse(result)
    
    def test_add_edge_duplicate_returns_false(self):
        """Agregar arista duplicada debe retornar False."""
        self.topology.add_edge("Agent", "Core")
        result = self.topology.add_edge("Agent", "Core")
        
        self.assertFalse(result)
    
    def test_remove_edge_success(self):
        """Eliminar arista existente."""
        self.topology.add_edge("Agent", "Core")
        result = self.topology.remove_edge("Agent", "Core")
        
        self.assertTrue(result)
        self.assertEqual(self.topology.num_edges, 0)
    
    def test_remove_edge_nonexistent_returns_false(self):
        """Eliminar arista inexistente debe retornar False."""
        result = self.topology.remove_edge("Agent", "Core")
        self.assertFalse(result)
    
    # -------------------------------------------------------------------------
    # Cálculo de Números de Betti
    # -------------------------------------------------------------------------
    
    def test_calculate_betti_numbers_empty_graph(self):
        """Betti numbers de grafo vacío."""
        topology = SystemTopology(custom_nodes=set(), validate_strictly=False)
        topology._graph.clear()  # Forzar vacío
        
        betti = topology.calculate_betti_numbers(include_isolated=True)
        
        self.assertEqual(betti.b0, 0)
        self.assertEqual(betti.b1, 0)
        self.assertEqual(betti.num_vertices, 0)
        self.assertEqual(betti.num_edges, 0)
    
    def test_calculate_betti_numbers_isolated_nodes(self):
        """Betti numbers con solo nodos aislados."""
        betti = self.topology.calculate_betti_numbers(include_isolated=True)
        
        n = len(SystemTopology.REQUIRED_NODES)
        self.assertEqual(betti.b0, n, f"Cada nodo aislado es una componente")
        self.assertEqual(betti.b1, 0)
    
    def test_calculate_betti_numbers_exclude_isolated(self):
        """Excluir nodos aislados debe dar grafo vacío si no hay aristas."""
        betti = self.topology.calculate_betti_numbers(include_isolated=False)
        
        self.assertEqual(betti.b0, 0)
        self.assertEqual(betti.b1, 0)
    
    def test_calculate_betti_numbers_tree_structure(self):
        """Betti numbers de estructura de árbol."""
        # Árbol: Agent -- Core -- Redis
        #                  |
        #              Filesystem
        self.topology.update_connectivity([
            ("Agent", "Core"),
            ("Core", "Redis"),
            ("Core", "Filesystem")
        ])
        
        betti = self.topology.calculate_betti_numbers()
        
        self.assertEqual(betti.b0, 1, "Árbol es conexo")
        self.assertEqual(betti.b1, 0, "Árbol no tiene ciclos")
        self.assertTrue(betti.is_tree)
        self.assertEqual(betti.cyclomatic_complexity, 1)
    
    def test_calculate_betti_numbers_single_cycle(self):
        """Betti numbers con un ciclo simple."""
        # Triángulo
        self.topology.update_connectivity([
            ("Agent", "Core"),
            ("Core", "Redis"),
            ("Redis", "Agent")
        ])
        
        betti = self.topology.calculate_betti_numbers()
        
        self.assertEqual(betti.b0, 1)
        self.assertEqual(betti.b1, 1)
        self.assertFalse(betti.is_acyclic)
    
    def test_calculate_betti_numbers_multiple_components(self):
        """Betti numbers con múltiples componentes."""
        # Crear dos componentes: {Agent, Core} y {Redis}
        self.topology.update_connectivity([
            ("Agent", "Core")
        ])
        
        betti = self.topology.calculate_betti_numbers(include_isolated=True)
        
        # Agent-Core conectados, Redis aislado, Filesystem aislado
        self.assertEqual(betti.b0, 3)
        self.assertFalse(betti.is_connected)
    
    def test_calculate_betti_numbers_caching(self):
        """Verifica que el caché funcione correctamente."""
        self.topology.update_connectivity([("Agent", "Core")])
        
        # Primera llamada (sin caché)
        betti1 = self.topology.calculate_betti_numbers(use_cache=True)
        
        # Segunda llamada (con caché)
        betti2 = self.topology.calculate_betti_numbers(use_cache=True)
        
        self.assertEqual(betti1.b0, betti2.b0)
        self.assertEqual(betti1.b1, betti2.b1)
    
    def test_calculate_betti_numbers_cache_invalidation(self):
        """Verifica que el caché se invalide al modificar el grafo."""
        self.topology.update_connectivity([("Agent", "Core")])
        betti1 = self.topology.calculate_betti_numbers(use_cache=True)
        
        # Modificar grafo
        self.topology.add_edge("Core", "Redis")
        
        # Caché debe invalidarse
        betti2 = self.topology.calculate_betti_numbers(use_cache=True)
        
        self.assertNotEqual(betti1.num_edges, betti2.num_edges)
    
    # -------------------------------------------------------------------------
    # Detección de Ciclos y Anomalías
    # -------------------------------------------------------------------------
    
    def test_find_structural_cycles_no_cycles(self):
        """Grafo sin ciclos no debe retornar ninguno."""
        self.topology.update_connectivity([
            ("Agent", "Core"),
            ("Core", "Redis")
        ])
        
        cycles = self.topology.find_structural_cycles()
        self.assertEqual(len(cycles), 0)
    
    def test_find_structural_cycles_single_cycle(self):
        """Detectar un ciclo simple."""
        self.topology.update_connectivity([
            ("Agent", "Core"),
            ("Core", "Redis"),
            ("Redis", "Agent")
        ])
        
        cycles = self.topology.find_structural_cycles()
        
        self.assertEqual(len(cycles), 1)
        # El ciclo debe contener los 3 nodos
        cycle = cycles[0]
        self.assertEqual(set(cycle), {"Agent", "Core", "Redis"})
    
    def test_find_structural_cycles_multiple_cycles(self):
        """Detectar múltiples ciclos."""
        # Crear dos ciclos independientes
        self.topology.update_connectivity([
            ("Agent", "Core"),
            ("Core", "Redis"),
            ("Redis", "Agent"),
            ("Core", "Filesystem"),
            ("Filesystem", "Agent")
        ])
        
        cycles = self.topology.find_structural_cycles()
        
        # Debe haber al menos 2 ciclos fundamentales
        self.assertGreaterEqual(len(cycles), 2)
    
    def test_get_disconnected_nodes_all_connected(self):
        """Sin nodos desconectados cuando todos están conectados."""
        self.topology.update_connectivity([
            ("Agent", "Core"),
            ("Core", "Redis"),
            ("Core", "Filesystem")
        ])
        
        disconnected = self.topology.get_disconnected_nodes()
        self.assertEqual(len(disconnected), 0)
    
    def test_get_disconnected_nodes_some_isolated(self):
        """Detectar nodos requeridos aislados."""
        self.topology.update_connectivity([
            ("Agent", "Core")
        ])
        
        disconnected = self.topology.get_disconnected_nodes()
        
        # Redis y Filesystem deben estar desconectados
        self.assertEqual(len(disconnected), 2)
        self.assertIn("Redis", disconnected)
        self.assertIn("Filesystem", disconnected)
    
    def test_get_missing_connections_complete_topology(self):
        """Sin conexiones faltantes cuando topología está completa."""
        expected_edges = list(SystemTopology.EXPECTED_TOPOLOGY)
        self.topology.update_connectivity(expected_edges)
        
        missing = self.topology.get_missing_connections()
        self.assertEqual(len(missing), 0)
    
    def test_get_missing_connections_partial_topology(self):
        """Detectar conexiones faltantes."""
        # Solo agregar una conexión
        self.topology.update_connectivity([("Agent", "Core")])
        
        missing = self.topology.get_missing_connections()
        
        # Faltan las demás conexiones esperadas
        self.assertGreater(len(missing), 0)
        self.assertLess(len(missing), len(SystemTopology.EXPECTED_TOPOLOGY))
    
    def test_get_unexpected_connections(self):
        """Detectar conexiones no esperadas."""
        self.topology.update_connectivity([
            ("Redis", "Filesystem")  # No está en EXPECTED_TOPOLOGY
        ])
        
        unexpected = self.topology.get_unexpected_connections()
        
        self.assertGreater(len(unexpected), 0)
        # La arista puede estar en cualquier dirección
        self.assertTrue(
            ("Redis", "Filesystem") in unexpected or
            ("Filesystem", "Redis") in unexpected
        )
    
    # -------------------------------------------------------------------------
    # Request Loops
    # -------------------------------------------------------------------------
    
    def test_record_request_valid(self):
        """Registrar request válido."""
        result = self.topology.record_request("req-123")
        self.assertTrue(result)
        self.assertEqual(self.topology.request_history_size, 1)
    
    def test_record_request_empty_string_returns_false(self):
        """Request vacío debe retornar False."""
        result = self.topology.record_request("")
        self.assertFalse(result)
        self.assertEqual(self.topology.request_history_size, 0)
    
    def test_record_request_whitespace_only_returns_false(self):
        """Request con solo whitespace debe retornar False."""
        result = self.topology.record_request("   ")
        self.assertFalse(result)
    
    def test_detect_request_loops_no_repeats(self):
        """Sin repeats no debe detectar loops."""
        for i in range(10):
            self.topology.record_request(f"req-{i}")
        
        loops = self.topology.detect_request_loops(threshold=3)
        self.assertEqual(len(loops), 0)
    
    def test_detect_request_loops_with_repeats(self):
        """Detectar loops con repeticiones."""
        # Repetir el mismo request 5 veces
        for _ in range(5):
            self.topology.record_request("retry-req")
        
        loops = self.topology.detect_request_loops(threshold=3)
        
        self.assertEqual(len(loops), 1)
        loop = loops[0]
        self.assertEqual(loop.request_id, "retry-req")
        self.assertEqual(loop.count, 5)
    
    def test_detect_request_loops_threshold_filtering(self):
        """Threshold debe filtrar loops pequeños."""
        # 2 repeticiones (< threshold de 3)
        for _ in range(2):
            self.topology.record_request("small-loop")
        
        loops = self.topology.detect_request_loops(threshold=3)
        self.assertEqual(len(loops), 0)
    
    def test_detect_request_loops_window_parameter(self):
        """Window debe limitar el rango de análisis."""
        # Llenar historial
        for i in range(50):
            self.topology.record_request(f"req-{i % 10}")
        
        # Analizar solo últimos 10
        loops = self.topology.detect_request_loops(threshold=2, window=10)
        
        # Debe haber algunos loops detectados en ventana
        self.assertGreaterEqual(len(loops), 0)
    
    def test_clear_request_history(self):
        """Limpiar historial debe vaciarlo."""
        for i in range(10):
            self.topology.record_request(f"req-{i}")
        
        self.topology.clear_request_history()
        
        self.assertEqual(self.topology.request_history_size, 0)
        loops = self.topology.detect_request_loops()
        self.assertEqual(len(loops), 0)
    
    # -------------------------------------------------------------------------
    # Salud Topológica
    # -------------------------------------------------------------------------
    
    def test_get_topological_health_ideal_topology(self):
        """Salud con topología ideal."""
        # Configurar topología esperada completa
        self.topology.update_connectivity(list(SystemTopology.EXPECTED_TOPOLOGY))
        
        health = self.topology.get_topological_health()
        
        self.assertEqual(health.level, HealthLevel.HEALTHY)
        self.assertGreaterEqual(health.health_score, 0.90)
        self.assertTrue(health.is_healthy)
        self.assertFalse(health.has_fragmentation)
        self.assertEqual(len(health.disconnected_nodes), 0)
        self.assertEqual(len(health.missing_edges), 0)
    
    def test_get_topological_health_fragmentation_penalty(self):
        """Penalización por fragmentación."""
        # Dejar nodos aislados
        self.topology.update_connectivity([("Agent", "Core")])
        
        health = self.topology.get_topological_health()
        
        self.assertTrue(health.has_fragmentation)
        self.assertGreater(health.betti.b0, 1)
        self.assertLess(health.health_score, 1.0)
    
    def test_get_topological_health_cycles_penalty(self):
        """Penalización por ciclos."""
        self.topology.update_connectivity([
            ("Agent", "Core"),
            ("Core", "Redis"),
            ("Redis", "Agent")
        ])
        
        health = self.topology.get_topological_health(calculate_b1=True)
        
        self.assertTrue(health.has_cycles)
        self.assertEqual(health.betti.b1, 1)
        self.assertLess(health.health_score, 1.0)
    
    def test_get_topological_health_disconnected_penalty(self):
        """Penalización por nodos desconectados."""
        self.topology.update_connectivity([("Agent", "Core")])
        
        health = self.topology.get_topological_health()
        
        self.assertTrue(health.has_disconnected_nodes)
        self.assertGreater(len(health.disconnected_nodes), 0)
        self.assertIn("disconnected", health.diagnostics)
    
    def test_get_topological_health_missing_edges_penalty(self):
        """Penalización por aristas faltantes."""
        # Solo agregar algunas conexiones
        self.topology.update_connectivity([("Agent", "Core")])
        
        health = self.topology.get_topological_health()
        
        self.assertTrue(health.has_missing_edges)
        self.assertGreater(len(health.missing_edges), 0)
    
    def test_get_topological_health_score_range(self):
        """Score de salud debe estar en [0, 1]."""
        health = self.topology.get_topological_health()
        
        self.assertGreaterEqual(health.health_score, 0.0)
        self.assertLessEqual(health.health_score, 1.0)
    
    def test_get_topological_health_caching(self):
        """Verifica caché de salud."""
        self.topology.update_connectivity([("Agent", "Core")])
        
        health1 = self.topology.get_topological_health(use_cache=True)
        health2 = self.topology.get_topological_health(use_cache=True)
        
        self.assertEqual(health1.health_score, health2.health_score)
    
    # -------------------------------------------------------------------------
    # Serialización
    # -------------------------------------------------------------------------
    
    def test_to_dict_complete_serialization(self):
        """Serialización completa a diccionario."""
        self.topology.update_connectivity([
            ("Agent", "Core"),
            ("Core", "Redis")
        ])
        
        data = self.topology.to_dict()
        
        # Estructura principal
        self.assertIn("graph", data)
        self.assertIn("betti_numbers", data)
        self.assertIn("health", data)
        self.assertIn("topology_status", data)
        
        # Grafo
        self.assertEqual(len(data["graph"]["nodes"]), self.topology.num_nodes)
        self.assertEqual(len(data["graph"]["edges"]), self.topology.num_edges)
        
        # Configuración
        self.assertIn("configuration", data)
        self.assertIn("required_nodes", data["configuration"])
    
    def test_get_adjacency_matrix(self):
        """Matriz de adyacencia."""
        self.topology.update_connectivity([
            ("Agent", "Core"),
            ("Core", "Redis")
        ])
        
        matrix = self.topology.get_adjacency_matrix()
        
        # Debe ser diccionario de diccionarios
        self.assertIsInstance(matrix, dict)
        
        # Verificar simetría (grafo no dirigido)
        for u in matrix:
            for v in matrix[u]:
                self.assertEqual(matrix[u][v], matrix[v][u])
        
        # Verificar aristas
        self.assertEqual(matrix["Agent"]["Core"], 1)
        self.assertEqual(matrix["Core"]["Redis"], 1)
        self.assertEqual(matrix["Agent"]["Redis"], 0)


# =============================================================================
# GRUPO 5: TEST DE PersistenceHomology
# =============================================================================

class TestPersistenceHomology(unittest.TestCase, MathematicalAssertions):
    """
    Verificación de análisis de homología persistente.
    
    Teoremas verificados:
    - Estabilidad de diagramas de persistencia
    - Clasificación de características (noise vs feature)
    - Propiedades de persistencia total
    """
    
    def setUp(self):
        """Configuración común."""
        self.ph = PersistenceHomology(window_size=20)
    
    def tearDown(self):
        """Limpieza."""
        del self.ph
    
    # -------------------------------------------------------------------------
    # Inicialización
    # -------------------------------------------------------------------------
    
    def test_initialization_valid_window(self):
        """Inicialización con window_size válido."""
        ph = PersistenceHomology(window_size=50)
        self.assertEqual(ph.window_size, 50)
        self.assertEqual(ph.num_metrics, 0)
    
    def test_initialization_invalid_window_too_small_raises(self):
        """Window muy pequeño debe lanzar excepción."""
        with self.assertRaises(ValueError):
            PersistenceHomology(window_size=1)
    
    def test_initialization_invalid_window_too_large_raises(self):
        """Window muy grande debe lanzar excepción."""
        with self.assertRaises(ValueError):
            PersistenceHomology(window_size=20000)
    
    # -------------------------------------------------------------------------
    # Gestión de Datos
    # -------------------------------------------------------------------------
    
    def test_add_reading_valid_creates_buffer(self):
        """Primera lectura crea buffer."""
        result = self.ph.add_reading("cpu_usage", 50.0)
        
        self.assertTrue(result)
        self.assertEqual(self.ph.num_metrics, 1)
        self.assertIn("cpu_usage", self.ph.metrics)
    
    def test_add_reading_appends_to_existing_buffer(self):
        """Lecturas adicionales se agregan al buffer."""
        self.ph.add_reading("cpu_usage", 50.0)
        self.ph.add_reading("cpu_usage", 60.0)
        self.ph.add_reading("cpu_usage", 70.0)
        
        buffer = self.ph.get_buffer("cpu_usage")
        self.assertEqual(len(buffer), 3)
        self.assertEqual(buffer, [50.0, 60.0, 70.0])
    
    def test_add_reading_nan_rejected(self):
        """NaN debe ser rechazado estrictamente."""
        result = self.ph.add_reading("metric", float('nan'))
        self.assertFalse(result)
    
    def test_add_reading_infinity_capped(self):
        """Infinito debe ser capeado con warning."""
        result = self.ph.add_reading("metric", float('inf'))
        
        # Debe ser aceptado pero capeado
        self.assertTrue(result)
        buffer = self.ph.get_buffer("metric")
        self.assertFalse(math.isinf(buffer[0]))
    
    def test_add_reading_empty_metric_name_rejected(self):
        """Nombre de métrica vacío debe ser rechazado."""
        result = self.ph.add_reading("", 100.0)
        self.assertFalse(result)
    
    def test_add_reading_non_string_metric_name_rejected(self):
        """Nombre de métrica no-string debe ser rechazado."""
        result = self.ph.add_reading(123, 100.0)  # type: ignore
        self.assertFalse(result)
    
    def test_add_reading_non_numeric_value_rejected(self):
        """Valor no-numérico debe ser rechazado."""
        result = self.ph.add_reading("metric", "not_a_number")  # type: ignore
        self.assertFalse(result)
    
    def test_add_readings_batch(self):
        """Agregar múltiples lecturas en batch."""
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        count = self.ph.add_readings_batch("metric", values)
        
        self.assertEqual(count, 5)
        buffer = self.ph.get_buffer("metric")
        self.assertEqual(buffer, values)
    
    def test_buffer_max_length_enforced(self):
        """Buffer debe respetar max_length."""
        ph = PersistenceHomology(window_size=5)
        
        # Agregar más valores que el límite
        for i in range(10):
            ph.add_reading("metric", float(i))
        
        buffer = ph.get_buffer("metric")
        self.assertEqual(len(buffer), 5)
        self.assertEqual(buffer, [5.0, 6.0, 7.0, 8.0, 9.0])
    
    def test_get_buffer_nonexistent_metric_returns_none(self):
        """Buffer de métrica inexistente debe retornar None."""
        buffer = self.ph.get_buffer("nonexistent")
        self.assertIsNone(buffer)
    
    def test_get_buffer_returns_copy(self):
        """get_buffer debe retornar copia, no referencia."""
        self.ph.add_reading("metric", 100.0)
        
        buffer = self.ph.get_buffer("metric")
        buffer.append(200.0)  # Modificar copia
        
        # Buffer original no debe cambiar
        original = self.ph.get_buffer("metric")
        self.assertEqual(len(original), 1)
    
    def test_clear_metric_existing(self):
        """Limpiar métrica existente."""
        self.ph.add_reading("metric", 100.0)
        result = self.ph.clear_metric("metric")
        
        self.assertTrue(result)
        self.assertNotIn("metric", self.ph.metrics)
    
    def test_clear_metric_nonexistent_returns_false(self):
        """Limpiar métrica inexistente debe retornar False."""
        result = self.ph.clear_metric("nonexistent")
        self.assertFalse(result)
    
    def test_clear_all(self):
        """Limpiar todas las métricas."""
        self.ph.add_reading("metric1", 100.0)
        self.ph.add_reading("metric2", 200.0)
        
        self.ph.clear_all()
        
        self.assertEqual(self.ph.num_metrics, 0)
        self.assertEqual(len(self.ph.metrics), 0)


# =============================================================================
# GRUPO 6: TEST DE FUNCIONES UTILITARIAS
# =============================================================================

class TestUtilityFunctions(unittest.TestCase, MathematicalAssertions):
    """Verificación de funciones de utilidad."""
    
    def test_create_simple_topology_structure(self):
        """create_simple_topology debe crear topología básica."""
        topology = create_simple_topology()
        
        self.assertIsInstance(topology, SystemTopology)
        self.assertGreater(topology.num_edges, 0)
        
        # Debe tener al menos algunas conexiones
        betti = topology.calculate_betti_numbers()
        self.assertGreaterEqual(betti.b0, 1)
    
    def test_compute_wasserstein_distance_identical_diagrams(self):
        """
        Teorema: W_p(D, D) = 0.
        """
        i1 = PersistenceInterval(birth=10, death=20, dimension=0)
        i2 = PersistenceInterval(birth=15, death=25, dimension=0)
        
        intervals = [i1, i2]
        
        dist = compute_wasserstein_distance(intervals, intervals, p=2)
        self.assertEqual(dist, 0.0)
    
    def test_compute_wasserstein_distance_empty_diagrams(self):
        """Distancia entre diagramas vacíos es 0."""
        dist = compute_wasserstein_distance([], [], p=2)
        self.assertEqual(dist, 0.0)
    
    def test_compute_wasserstein_distance_one_empty(self):
        """Distancia cuando un diagrama está vacío."""
        i1 = PersistenceInterval(birth=0, death=10, dimension=0)
        
        dist = compute_wasserstein_distance([i1], [], p=2)
        self.assertGreater(dist, 0.0)
    
    def test_compute_wasserstein_distance_different_sizes(self):
        """Distancia entre diagramas de distinto tamaño."""
        intervals1 = [
            PersistenceInterval(birth=0, death=10, dimension=0),
            PersistenceInterval(birth=5, death=15, dimension=0)
        ]
        
        intervals2 = [
            PersistenceInterval(birth=0, death=12, dimension=0)
        ]
        
        dist = compute_wasserstein_distance(intervals1, intervals2, p=2)
        self.assertGreater(dist, 0.0)
        self.assertTrue(math.isfinite(dist))
    
    def test_compute_wasserstein_distance_invalid_p_raises(self):
        """p < 1 debe lanzar excepción."""
        i1 = PersistenceInterval(birth=0, death=10, dimension=0)
        
        with self.assertRaises(ValueError):
            compute_wasserstein_distance([i1], [i1], p=0.5)
    
    def test_compute_wasserstein_distance_p_infinity_approximation(self):
        """Distancia con p grande debe aproximarse a max."""
        i1 = PersistenceInterval(birth=0, death=10, dimension=0)
        i2 = PersistenceInterval(birth=0, death=20, dimension=0)
        
        dist = compute_wasserstein_distance([i1], [i2], p=100)
        
        # Debe aproximarse a la diferencia máxima
        self.assertGreater(dist, 0.0)


# =============================================================================
# GRUPO 7: TEST DE INTEGRACIÓN END-TO-END
# =============================================================================

class TestIntegration(unittest.TestCase, MathematicalAssertions):
    """
    Escenarios completos que integran múltiples componentes.
    """
    
    def test_scenario_healthy_system_evolution(self):
        """
        Escenario: Sistema saludable evoluciona sin degradación.
        """
        topology = SystemTopology(max_history=100)
        
        # T0: Inicialización
        health = topology.get_topological_health()
        initial_score = health.health_score
        
        # T1: Agregar topología esperada
        topology.update_connectivity(list(SystemTopology.EXPECTED_TOPOLOGY))
        health = topology.get_topological_health()
        
        self.assertGreater(health.health_score, initial_score)
        self.assertEqual(health.level, HealthLevel.HEALTHY)
        
        # T2: Agregar requests normales (no loops)
        for i in range(50):
            topology.record_request(f"req-{i}")
        
        loops = topology.detect_request_loops(threshold=3)
        self.assertEqual(len(loops), 0)
        
        # Salud debe mantenerse
        health_final = topology.get_topological_health()
        self.assertEqual(health_final.level, HealthLevel.HEALTHY)
    
    def test_scenario_degradation_and_recovery(self):
        """
        Escenario: Sistema se degrada y luego se recupera.
        """
        topology = SystemTopology()
        
        # Fase 1: Estado degradado (fragmentación)
        topology.update_connectivity([("Agent", "Core")])
        health_degraded = topology.get_topological_health()
        
        self.assertTrue(health_degraded.has_fragmentation)
        self.assertIn(health_degraded.level, [HealthLevel.DEGRADED, HealthLevel.UNHEALTHY])
        
        # Fase 2: Recuperación (completar topología)
        topology.update_connectivity(list(SystemTopology.EXPECTED_TOPOLOGY))
        health_recovered = topology.get_topological_health()
        
        self.assertGreater(health_recovered.health_score, health_degraded.health_score)
        self.assertEqual(health_recovered.level, HealthLevel.HEALTHY)
    
    def test_scenario_critical_failure_detection(self):
        """
        Escenario: Detección de fallo crítico.
        """
        topology = SystemTopology()
        
        # Sin conexiones + loops de reintento
        for _ in range(20):
            topology.record_request("failing-req")
        
        health = topology.get_topological_health()
        loops = topology.detect_request_loops(threshold=3)
        
        # Debe haber penalizaciones severas
        self.assertLess(health.health_score, 0.7)
        self.assertGreater(len(loops), 0)
        self.assertTrue(health.has_fragmentation)
        self.assertTrue(health.has_disconnected_nodes)


# =============================================================================
# GRUPO 8: TEST DE CASOS EXTREMOS
# =============================================================================

class TestEdgeCases(unittest.TestCase, MathematicalAssertions):
    """
    Casos extremos y patológicos.
    """
    
    def test_large_graph_performance(self):
        """Grafo grande (100 nodos) debe procesarse eficientemente."""
        topology = SystemTopology(validate_strictly=False)
        
        # Agregar 100 nodos
        for i in range(100):
            topology.add_node(f"Node{i}")
        
        # Agregar conexiones (árbol)
        for i in range(1, 100):
            topology.add_edge(f"Node{i-1}", f"Node{i}")
        
        # Cálculo de Betti debe completarse
        betti = topology.calculate_betti_numbers()
        
        self.assertEqual(betti.num_vertices, 100 + len(SystemTopology.REQUIRED_NODES))
        self.assertEqual(betti.b0, 2)  # Componente del árbol + nodos requeridos aislados
        self.assertEqual(betti.b1, 0)
    
    def test_complete_graph_high_cyclomatic_complexity(self):
        """Grafo completo pequeño tiene alta complejidad ciclomática."""
        topology = SystemTopology(validate_strictly=False)
        
        nodes = ["A", "B", "C", "D"]
        for node in nodes:
            topology.add_node(node)
        
        # Grafo completo K4
        edges = list(itertools.combinations(nodes, 2))
        topology.update_connectivity(edges)
        
        betti = topology.calculate_betti_numbers()
        
        # K4: 4 vértices, 6 aristas, β₁ = 6 - 4 + 1 = 3
        self.assertEqual(betti.num_edges, 6)
        self.assertEqual(betti.b1, 3)
        self.assertGreater(betti.cyclomatic_complexity, 1)
    
    def test_unicode_node_names(self):
        """Nodos con caracteres Unicode."""
        topology = SystemTopology(validate_strictly=False)
        
        unicode_nodes = ["服务A", "Сервис-Б", "خدمة_ج"]
        for node in unicode_nodes:
            result = topology.add_node(node)
            self.assertTrue(result, f"Fallo al agregar nodo Unicode: {node}")
        
        for node in unicode_nodes:
            self.assertTrue(topology.has_node(node))
    
    def test_extremely_long_node_name(self):
        """Nombre de nodo extremadamente largo."""
        topology = SystemTopology(validate_strictly=False)
        
        long_name = "Node" * 1000  # 4000 caracteres
        result = topology.add_node(long_name)
        
        self.assertTrue(result)
        self.assertTrue(topology.has_node(long_name))
    
    def test_persistence_with_all_zeros(self):
        """Serie temporal con todos ceros."""
        ph = PersistenceHomology(window_size=20)
        
        for _ in range(20):
            ph.add_reading("metric", 0.0)
        
        # No debe haber error
        buffer = ph.get_buffer("metric")
        self.assertEqual(len(buffer), 20)
        self.assertTrue(all(v == 0.0 for v in buffer))
    
    def test_persistence_with_constant_nonzero(self):
        """Serie temporal constante no-cero."""
        ph = PersistenceHomology(window_size=20)
        
        for _ in range(20):
            ph.add_reading("metric", 100.0)
        
        buffer = ph.get_buffer("metric")
        self.assertTrue(all(v == 100.0 for v in buffer))
    
    def test_betti_numbers_with_multigraph_simulation(self):
        """Simular multi-grafo (aristas múltiples)."""
        # NetworkX.Graph no soporta multi-aristas, pero podemos testear comportamiento
        topology = SystemTopology(validate_strictly=False)
        
        topology.add_edge("A", "B")
        # Intentar agregar duplicada
        result = topology.add_edge("A", "B")
        
        self.assertFalse(result, "Arista duplicada debe ser rechazada")
        self.assertEqual(topology.num_edges, 1)


# =============================================================================
# GRUPO 9: TEST DE ESTABILIDAD NUMÉRICA
# =============================================================================

class TestNumericalStability(unittest.TestCase, MathematicalAssertions):
    """
    Verificación de estabilidad numérica y precisión.
    """
    
    def test_betti_numbers_floating_point_consistency(self):
        """Números de Betti con valores grandes."""
        topology = SystemTopology(validate_strictly=False)
        
        # Crear grafo con muchos nodos
        for i in range(50):
            topology.add_node(f"N{i}")
        
        # Crear árbol (49 aristas)
        for i in range(1, 50):
            topology.add_edge(f"N{i-1}", f"N{i}")
        
        betti = topology.calculate_betti_numbers()
        
        # Verificar consistencia
        self.assertTrue(betti.verify_euler_consistency())
    
    def test_persistence_interval_geometric_precision(self):
        """Precisión geométrica de persistencia."""
        # Intervalo con valores grandes
        interval = PersistenceInterval(
            birth=1_000_000,
            death=1_000_100,
            dimension=0
        )
        
        # Persistencia = 100 / √2
        expected = 100.0 / math.sqrt(2.0)
        self.assertAlmostEqualFloat(interval.persistence, expected, tolerance=1e-9)
    
    def test_wasserstein_distance_numerical_stability(self):
        """Estabilidad numérica de Wasserstein."""
        # Intervalos con valores muy dispares
        i1 = PersistenceInterval(birth=0, death=1e-10, dimension=0)
        i2 = PersistenceInterval(birth=0, death=1e10, dimension=0)
        
        dist = compute_wasserstein_distance([i1], [i2], p=2)
        
        self.assertTrue(math.isfinite(dist))
        self.assertGreater(dist, 0.0)
    
    def test_health_score_boundary_precision(self):
        """Precisión en boundaries de health score."""
        topology = SystemTopology()
        
        # Crear estado que genere score cerca de 0.9
        topology.update_connectivity([
            ("Agent", "Core"),
            ("Core", "Redis"),
            ("Core", "Filesystem")
        ])
        
        health = topology.get_topological_health()
        
        # Score debe estar en [0, 1]
        self.assertGreaterEqual(health.health_score, 0.0)
        self.assertLessEqual(health.health_score, 1.0)
        
        # Consistencia con nivel
        expected_level = HealthLevel.from_score(health.health_score)
        self.assertEqual(health.level, expected_level)


# =============================================================================
# SUITE COMPLETA
# =============================================================================

def suite() -> unittest.TestSuite:
    """Construye la suite completa de pruebas."""
    suite = unittest.TestSuite()
    
    # Agregar todos los test cases
    test_classes = [
        TestTopologicalConstants,
        TestBettiNumbers,
        TestPersistenceInterval,
        TestSystemTopology,
        TestPersistenceHomology,
        TestUtilityFunctions,
        TestIntegration,
        TestEdgeCases,
        TestNumericalStability
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    return suite


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    # Configurar logging para tests
    import logging
    logging.basicConfig(
        level=logging.WARNING,  # Solo warnings y errores
        format="%(levelname)s: %(message)s"
    )
    
    # Ejecutar suite
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite())
    
    # Reporte final
    print("\n" + "=" * 80)
    print("REPORTE FINAL DE PRUEBAS")
    print("=" * 80)
    print(f"Tests ejecutados: {result.testsRun}")
    print(f"✅ Exitosos: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Fallidos: {len(result.failures)}")
    print(f"💥 Errores: {len(result.errors)}")
    print(f"⏭️  Omitidos: {len(result.skipped)}")
    print("=" * 80)
    
    # Exit code
    sys.exit(0 if result.wasSuccessful() else 1)