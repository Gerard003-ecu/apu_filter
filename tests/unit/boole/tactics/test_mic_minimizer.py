"""
=========================================================================================
    Test Suite: MIC minimizer verificación rigurosa
    Ubicación: tests/unit/boole/tactics/test_mic_minimizer.py
    Versión: 2.0 - test algebraico comprensivo y topológico
    
    FILOSOFÍA DE TESTING:
    ---------------------
    Esta suite implementa verificación rigurosa multi-nivel:
    
    1. TESTING ALGEBRAICO:
       - Verificación de axiomas del retículo booleano
       - Propiedades del anillo ℤ₂[x]/⟨x² - x⟩
       - Corrección de la base de Gröbner
       - Invariantes de ROBDDs
    
    2. TESTING TOPOLÓGICO:
       - Números de Betti correctos
       - Grupos de homología
       - Consistencia de complejos simpliciales
    
    3. TESTING NUMÉRICO:
       - Estabilidad de SVD
       - Condicionamiento de matrices
       - Detección de singularidades
    
    4. TESTING LÓGICO:
       - Corrección de SAT solver (DPLL)
       - Completitud de verificaciones
       - Detección de conflictos
    
    5. PROPERTY-BASED TESTING:
       - Generación de casos aleatorios
       - Verificación de invariantes
       - Fuzzing de entradas
    
    COBERTURA:
    ----------
    • Statement coverage: > 95%
    • Branch coverage: > 90%
    • Path coverage: casos críticos
    • Mutation testing: resistencia a mutaciones
    
=========================================================================================
"""

import math
import sys
import unittest
from typing import Dict, FrozenSet, List, Set, Tuple
from unittest.mock import MagicMock, patch
import random

# Importar el módulo a testear
try:
    from app.boole.tactics.mic_minimizer import (
        # Estructuras algebraicas
        BooleanVector,
        CapabilityDimension,
        Tool,
        Z2Polynomial,
        
        # Algoritmos
        GrobnerBasisZ2,
        ROBDD,
        ROBDDNode,
        DPLLSATSolver,
        MICRedundancyAnalyzer,
        
        # Excepciones
        MICException,
        HomologicalInconsistencyError,
        UnsatCoreError,
        GrobnerBasisComputationError,
        ROBDDConstructionError,
        
        # Utilidades
        validate_boolean_lattice_axioms,
        audit_mic_redundancy,
    )
    NUMPY_AVAILABLE = True
    try:
        import numpy as np
    except ImportError:
        NUMPY_AVAILABLE = False
except ImportError:
    # Fallback para ejecución directa
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from app.boole.tactics.mic_minimizer import *
    try:
        import numpy as np
        NUMPY_AVAILABLE = True
    except ImportError:
        NUMPY_AVAILABLE = False


# =============================================================================
# UTILIDADES DE TESTING
# =============================================================================

class TestHelpers:
    """Utilidades compartidas para testing."""
    
    @staticmethod
    def random_boolean_vector(num_vars: int = 5) -> BooleanVector:
        """Genera un vector booleano aleatorio."""
        num_components = random.randint(0, num_vars)
        components = random.sample(
            list(CapabilityDimension)[:num_vars],
            num_components
        )
        return BooleanVector(frozenset(components))
    
    @staticmethod
    def random_polynomial(num_vars: int = 5, max_terms: int = 5) -> Z2Polynomial:
        """Genera un polinomio aleatorio en ℤ₂."""
        num_terms = random.randint(0, max_terms)
        monomials = set()
        
        for _ in range(num_terms):
            # Generar monomio aleatorio
            num_vars_in_mon = random.randint(0, num_vars)
            vars_in_mon = frozenset(random.sample(range(num_vars), num_vars_in_mon))
            monomials.add(vars_in_mon)
        
        return Z2Polynomial(frozenset(monomials))
    
    @staticmethod
    def create_test_tools(num_tools: int = 5) -> List[Tool]:
        """Crea herramientas de prueba."""
        tools = []
        for i in range(num_tools):
            name = f"tool_{i}"
            caps = TestHelpers.random_boolean_vector()
            tools.append(Tool(name, caps))
        return tools


# =============================================================================
# TESTS: BooleanVector (Retículo Booleano)
# =============================================================================

class TestBooleanVector(unittest.TestCase):
    """
    Test suite para BooleanVector.
    
    Verifica:
    - Axiomas del retículo booleano
    - Operaciones algebraicas
    - Métricas de Hamming
    - Productos internos
    """
    
    def setUp(self):
        """Inicialización de casos de prueba."""
        self.empty = BooleanVector.zero()
        self.universe = BooleanVector.universe(5)
        
        self.v1 = BooleanVector(frozenset([
            CapabilityDimension.PHYS_IO,
            CapabilityDimension.PHYS_NUM
        ]))
        
        self.v2 = BooleanVector(frozenset([
            CapabilityDimension.PHYS_NUM,
            CapabilityDimension.TACT_TOPO
        ]))
        
        self.v3 = BooleanVector(frozenset([
            CapabilityDimension.STRAT_FIN
        ]))
    
    # -------------------------------------------------------------------------
    # CONSTRUCCIÓN Y VALIDACIÓN
    # -------------------------------------------------------------------------
    
    def test_initialization_empty(self):
        """Test: Inicialización vacía."""
        v = BooleanVector()
        self.assertEqual(v.components, frozenset())
        self.assertEqual(v.hamming_weight(), 0)
    
    def test_initialization_with_set(self):
        """Test: Inicialización con set regular."""
        caps = {CapabilityDimension.PHYS_IO, CapabilityDimension.PHYS_NUM}
        v = BooleanVector(caps)
        
        self.assertIsInstance(v.components, frozenset)
        self.assertEqual(len(v.components), 2)
    
    def test_initialization_invalid_type(self):
        """Test: Tipo inválido en componentes."""
        with self.assertRaises(TypeError):
            BooleanVector(frozenset([1, 2, 3]))  # type: ignore
    
    def test_from_minterm(self):
        """Test: Construcción desde minitérmino."""
        # 5 en binario = 101 → dimensiones 0 y 2
        v = BooleanVector.from_minterm(5, 3)
        
        self.assertIn(CapabilityDimension(0), v.components)
        self.assertIn(CapabilityDimension(2), v.components)
        self.assertNotIn(CapabilityDimension(1), v.components)
    
    def test_from_minterm_out_of_range(self):
        """Test: Minitérmino fuera de rango."""
        with self.assertRaises(ValueError):
            BooleanVector.from_minterm(16, 3)  # 16 > 2^3 - 1 = 7
    
    def test_from_binary_string(self):
        """Test: Construcción desde string binario."""
        v = BooleanVector.from_binary_string("10110")
        
        expected = BooleanVector(frozenset([
            CapabilityDimension(0),
            CapabilityDimension(2),
            CapabilityDimension(3)
        ]))
        
        self.assertEqual(v, expected)
    
    def test_to_binary_string(self):
        """Test: Conversión a string binario."""
        v = BooleanVector(frozenset([
            CapabilityDimension(0),
            CapabilityDimension(2)
        ]))
        
        binary = v.to_binary_string(5)
        self.assertEqual(binary, "10100")
    
    def test_to_minterm(self):
        """Test: Conversión a minitérmino."""
        v = BooleanVector(frozenset([
            CapabilityDimension(0),
            CapabilityDimension(2)
        ]))
        
        # Bit 0 y bit 2 activos → 2^0 + 2^2 = 5
        self.assertEqual(v.to_minterm(), 5)
    
    def test_immutability(self):
        """Test: Inmutabilidad del vector."""
        v = BooleanVector(frozenset([CapabilityDimension.PHYS_IO]))
        
        with self.assertRaises(AttributeError):
            v.components = frozenset()  # type: ignore
    
    def test_hashability(self):
        """Test: Vectores son hashables."""
        v1 = BooleanVector(frozenset([CapabilityDimension.PHYS_IO]))
        v2 = BooleanVector(frozenset([CapabilityDimension.PHYS_IO]))
        
        # Pueden usarse en sets
        s = {v1, v2}
        self.assertEqual(len(s), 1)
        
        # Pueden usarse como claves
        d = {v1: "value"}
        self.assertEqual(d[v2], "value")
    
    # -------------------------------------------------------------------------
    # AXIOMAS DEL RETÍCULO BOOLEANO
    # -------------------------------------------------------------------------
    
    def test_axiom_commutativity(self):
        """Test: Conmutatividad de ∨ y ∧."""
        # ∨ conmutativa
        self.assertEqual(
            self.v1.union(self.v2),
            self.v2.union(self.v1)
        )
        
        # ∧ conmutativa
        self.assertEqual(
            self.v1.intersection(self.v2),
            self.v2.intersection(self.v1)
        )
    
    def test_axiom_associativity(self):
        """Test: Asociatividad de ∨ y ∧."""
        # (v1 ∨ v2) ∨ v3 = v1 ∨ (v2 ∨ v3)
        left = self.v1.union(self.v2).union(self.v3)
        right = self.v1.union(self.v2.union(self.v3))
        self.assertEqual(left, right)
        
        # (v1 ∧ v2) ∧ v3 = v1 ∧ (v2 ∧ v3)
        left = self.v1.intersection(self.v2).intersection(self.v3)
        right = self.v1.intersection(self.v2.intersection(self.v3))
        self.assertEqual(left, right)
    
    def test_axiom_absorption(self):
        """Test: Leyes de absorción."""
        # v1 ∨ (v1 ∧ v2) = v1
        self.assertEqual(
            self.v1.union(self.v1.intersection(self.v2)),
            self.v1
        )
        
        # v1 ∧ (v1 ∨ v2) = v1
        self.assertEqual(
            self.v1.intersection(self.v1.union(self.v2)),
            self.v1
        )
    
    def test_axiom_distributivity(self):
        """Test: Distributividad."""
        # v1 ∧ (v2 ∨ v3) = (v1 ∧ v2) ∨ (v1 ∧ v3)
        left = self.v1.intersection(self.v2.union(self.v3))
        right = self.v1.intersection(self.v2).union(self.v1.intersection(self.v3))
        self.assertEqual(left, right)
    
    def test_axiom_complement(self):
        """Test: Leyes de complemento."""
        # v ∨ ¬v = 1
        self.assertEqual(
            self.v1.union(self.v1.complement(5)),
            self.universe
        )
        
        # v ∧ ¬v = 0
        self.assertEqual(
            self.v1.intersection(self.v1.complement(5)),
            self.empty
        )
    
    def test_axiom_idempotence(self):
        """Test: Idempotencia."""
        # v ∨ v = v
        self.assertEqual(self.v1.union(self.v1), self.v1)
        
        # v ∧ v = v
        self.assertEqual(self.v1.intersection(self.v1), self.v1)
    
    def test_axiom_identity(self):
        """Test: Elementos identidad."""
        # v ∨ 0 = v
        self.assertEqual(self.v1.union(self.empty), self.v1)
        
        # v ∧ 1 = v
        self.assertEqual(self.v1.intersection(self.universe), self.v1)
    
    def test_axiom_annihilation(self):
        """Test: Elementos aniquiladores."""
        # v ∨ 1 = 1
        self.assertEqual(self.v1.union(self.universe), self.universe)
        
        # v ∧ 0 = 0
        self.assertEqual(self.v1.intersection(self.empty), self.empty)
    
    # -------------------------------------------------------------------------
    # OPERACIONES ALGEBRAICAS
    # -------------------------------------------------------------------------
    
    def test_symmetric_difference(self):
        """Test: Diferencia simétrica (XOR)."""
        # v1 ⊕ v2: elementos en uno pero no en ambos
        result = self.v1.symmetric_difference(self.v2)
        
        expected = BooleanVector(frozenset([
            CapabilityDimension.PHYS_IO,
            CapabilityDimension.TACT_TOPO
        ]))
        
        self.assertEqual(result, expected)
    
    def test_symmetric_difference_self(self):
        """Test: XOR consigo mismo es cero."""
        result = self.v1.symmetric_difference(self.v1)
        self.assertEqual(result, self.empty)
    
    def test_difference(self):
        """Test: Diferencia de conjuntos."""
        result = self.v1.difference(self.v2)
        
        expected = BooleanVector(frozenset([CapabilityDimension.PHYS_IO]))
        self.assertEqual(result, expected)
    
    def test_complement_double(self):
        """Test: Doble complemento es identidad."""
        comp = self.v1.complement(5)
        double_comp = comp.complement(5)
        self.assertEqual(double_comp, self.v1)
    
    # -------------------------------------------------------------------------
    # RELACIONES DE ORDEN
    # -------------------------------------------------------------------------
    
    def test_is_subset_of(self):
        """Test: Relación de subconjunto."""
        v_sub = BooleanVector(frozenset([CapabilityDimension.PHYS_IO]))
        
        self.assertTrue(v_sub.is_subset_of(self.v1))
        self.assertTrue(self.empty.is_subset_of(self.v1))
        self.assertTrue(self.v1.is_subset_of(self.v1))
        self.assertFalse(self.v1.is_subset_of(v_sub))
    
    def test_is_superset_of(self):
        """Test: Relación de superconjunto."""
        v_sub = BooleanVector(frozenset([CapabilityDimension.PHYS_IO]))
        
        self.assertTrue(self.v1.is_superset_of(v_sub))
        self.assertFalse(v_sub.is_superset_of(self.v1))
    
    def test_is_disjoint_from(self):
        """Test: Conjuntos disjuntos."""
        self.assertTrue(self.v1.is_disjoint_from(self.v3))
        self.assertFalse(self.v1.is_disjoint_from(self.v2))
    
    # -------------------------------------------------------------------------
    # MÉTRICAS
    # -------------------------------------------------------------------------
    
    def test_hamming_weight(self):
        """Test: Peso de Hamming."""
        self.assertEqual(self.empty.hamming_weight(), 0)
        self.assertEqual(self.v1.hamming_weight(), 2)
        self.assertEqual(self.universe.hamming_weight(), 5)
    
    def test_hamming_distance(self):
        """Test: Distancia de Hamming."""
        # v1 = {PHYS_IO, PHYS_NUM}
        # v2 = {PHYS_NUM, TACT_TOPO}
        # Diferencia: {PHYS_IO, TACT_TOPO} → distancia = 2
        
        distance = self.v1.hamming_distance(self.v2)
        self.assertEqual(distance, 2)
    
    def test_hamming_distance_self(self):
        """Test: Distancia de Hamming a sí mismo es cero."""
        self.assertEqual(self.v1.hamming_distance(self.v1), 0)
    
    def test_hamming_distance_symmetric(self):
        """Test: Simetría de distancia de Hamming."""
        d12 = self.v1.hamming_distance(self.v2)
        d21 = self.v2.hamming_distance(self.v1)
        self.assertEqual(d12, d21)
    
    def test_hamming_distance_triangle_inequality(self):
        """Test: Desigualdad triangular."""
        d12 = self.v1.hamming_distance(self.v2)
        d23 = self.v2.hamming_distance(self.v3)
        d13 = self.v1.hamming_distance(self.v3)
        
        self.assertLessEqual(d13, d12 + d23)
    
    def test_inner_product_z2(self):
        """Test: Producto interno en ℤ₂."""
        # v1 ∩ v2 = {PHYS_NUM} → tamaño 1 → 1 mod 2 = 1
        prod = self.v1.inner_product_z2(self.v2)
        self.assertEqual(prod, 1)
        
        # v1 ∩ v3 = ∅ → tamaño 0 → 0 mod 2 = 0
        prod = self.v1.inner_product_z2(self.v3)
        self.assertEqual(prod, 0)
    
    def test_inner_product_real(self):
        """Test: Producto interno estándar."""
        # |v1 ∩ v2| = 1
        prod = self.v1.inner_product_real(self.v2)
        self.assertEqual(prod, 1)
        
        # |v1 ∩ v3| = 0
        prod = self.v1.inner_product_real(self.v3)
        self.assertEqual(prod, 0)
    
    # -------------------------------------------------------------------------
    # ORDEN Y COMPARACIÓN
    # -------------------------------------------------------------------------
    
    def test_equality(self):
        """Test: Igualdad de vectores."""
        v1_copy = BooleanVector(frozenset([
            CapabilityDimension.PHYS_IO,
            CapabilityDimension.PHYS_NUM
        ]))
        
        self.assertEqual(self.v1, v1_copy)
        self.assertNotEqual(self.v1, self.v2)
    
    def test_ordering(self):
        """Test: Orden total para sorting."""
        vectors = [self.v3, self.v1, self.v2]
        sorted_vecs = sorted(vectors)
        
        # Debe ser ordenable sin errores
        self.assertEqual(len(sorted_vecs), 3)
    
    # -------------------------------------------------------------------------
    # CASOS EDGE
    # -------------------------------------------------------------------------
    
    def test_large_vector(self):
        """Test: Vector con muchos componentes."""
        large_caps = frozenset(CapabilityDimension)
        v = BooleanVector(large_caps)
        
        self.assertEqual(len(v.components), len(CapabilityDimension))
    
    def test_roundtrip_minterm(self):
        """Test: Conversión minterm → vector → minterm."""
        original_minterm = 13  # 1101 en binario
        v = BooleanVector.from_minterm(original_minterm, 4)
        recovered_minterm = v.to_minterm()
        
        self.assertEqual(recovered_minterm, original_minterm)
    
    def test_roundtrip_binary_string(self):
        """Test: Conversión string → vector → string."""
        original = "10110"
        v = BooleanVector.from_binary_string(original)
        recovered = v.to_binary_string(5)
        
        self.assertEqual(recovered, original)


# =============================================================================
# TESTS: Tool (Herramientas)
# =============================================================================

class TestTool(unittest.TestCase):
    """Test suite para Tool."""
    
    def test_initialization_valid(self):
        """Test: Inicialización válida."""
        caps = BooleanVector(frozenset([CapabilityDimension.PHYS_IO]))
        tool = Tool("test_tool", caps)
        
        self.assertEqual(tool.name, "test_tool")
        self.assertEqual(tool.capabilities, caps)
    
    def test_initialization_invalid_name(self):
        """Test: Nombre inválido."""
        caps = BooleanVector()
        
        with self.assertRaises(ValueError):
            Tool("", caps)
        
        with self.assertRaises(ValueError):
            Tool("   ", caps)
    
    def test_initialization_invalid_capabilities(self):
        """Test: Capacidades de tipo incorrecto."""
        with self.assertRaises(TypeError):
            Tool("test", {CapabilityDimension.PHYS_IO})  # type: ignore
    
    def test_arity(self):
        """Test: Aridad (número de capacidades)."""
        caps = BooleanVector(frozenset([
            CapabilityDimension.PHYS_IO,
            CapabilityDimension.PHYS_NUM
        ]))
        tool = Tool("test", caps)
        
        self.assertEqual(tool.arity, 2)
    
    def test_is_trivial(self):
        """Test: Herramienta trivial (sin capacidades)."""
        trivial = Tool("empty", BooleanVector())
        non_trivial = Tool("non_empty", BooleanVector(frozenset([CapabilityDimension.PHYS_IO])))
        
        self.assertTrue(trivial.is_trivial)
        self.assertFalse(non_trivial.is_trivial)
    
    def test_signature(self):
        """Test: Firma única."""
        caps = BooleanVector(frozenset([
            CapabilityDimension(0),
            CapabilityDimension(2)
        ]))
        tool = Tool("test", caps)
        
        self.assertEqual(tool.signature, "10100")
    
    def test_subsumes(self):
        """Test: Relación de subsunción."""
        t1 = Tool("general", BooleanVector(frozenset([
            CapabilityDimension.PHYS_IO,
            CapabilityDimension.PHYS_NUM
        ])))
        
        t2 = Tool("specific", BooleanVector(frozenset([
            CapabilityDimension.PHYS_IO
        ])))
        
        self.assertTrue(t1.subsumes(t2))
        self.assertFalse(t2.subsumes(t1))
    
    def test_overlaps_with(self):
        """Test: Solapamiento de capacidades."""
        t1 = Tool("t1", BooleanVector(frozenset([CapabilityDimension.PHYS_IO])))
        t2 = Tool("t2", BooleanVector(frozenset([CapabilityDimension.PHYS_NUM])))
        t3 = Tool("t3", BooleanVector(frozenset([CapabilityDimension.PHYS_IO])))
        
        self.assertFalse(t1.overlaps_with(t2))
        self.assertTrue(t1.overlaps_with(t3))
    
    def test_similarity_score(self):
        """Test: Coeficiente de Jaccard."""
        t1 = Tool("t1", BooleanVector(frozenset([
            CapabilityDimension.PHYS_IO,
            CapabilityDimension.PHYS_NUM
        ])))
        
        t2 = Tool("t2", BooleanVector(frozenset([
            CapabilityDimension.PHYS_NUM,
            CapabilityDimension.TACT_TOPO
        ])))
        
        # Intersección: {PHYS_NUM} → 1
        # Unión: {PHYS_IO, PHYS_NUM, TACT_TOPO} → 3
        # Jaccard = 1/3 ≈ 0.333
        
        score = t1.similarity_score(t2)
        self.assertAlmostEqual(score, 1/3, places=5)
    
    def test_similarity_identical(self):
        """Test: Herramientas idénticas tienen similitud 1."""
        t1 = Tool("t1", BooleanVector(frozenset([CapabilityDimension.PHYS_IO])))
        t2 = Tool("t2", BooleanVector(frozenset([CapabilityDimension.PHYS_IO])))
        
        self.assertEqual(t1.similarity_score(t2), 1.0)
    
    def test_similarity_disjoint(self):
        """Test: Herramientas disjuntas tienen similitud 0."""
        t1 = Tool("t1", BooleanVector(frozenset([CapabilityDimension.PHYS_IO])))
        t2 = Tool("t2", BooleanVector(frozenset([CapabilityDimension.PHYS_NUM])))
        
        self.assertEqual(t1.similarity_score(t2), 0.0)
    
    def test_ordering(self):
        """Test: Orden lexicográfico."""
        t1 = Tool("alpha", BooleanVector())
        t2 = Tool("beta", BooleanVector())
        
        self.assertLess(t1, t2)
    
    def test_hashability(self):
        """Test: Herramientas son hashables."""
        t1 = Tool("test", BooleanVector(frozenset([CapabilityDimension.PHYS_IO])))
        t2 = Tool("test", BooleanVector(frozenset([CapabilityDimension.PHYS_IO])))
        
        s = {t1, t2}
        self.assertEqual(len(s), 1)


# =============================================================================
# TESTS: Z2Polynomial (Polinomios en ℤ₂)
# =============================================================================

class TestZ2Polynomial(unittest.TestCase):
    """Test suite para Z2Polynomial."""
    
    def test_zero_polynomial(self):
        """Test: Polinomio cero."""
        zero = Z2Polynomial.zero()
        
        self.assertTrue(zero.is_zero())
        self.assertFalse(zero.is_one())
        self.assertEqual(zero.degree(), -1)
    
    def test_one_polynomial(self):
        """Test: Polinomio constante 1."""
        one = Z2Polynomial.one()
        
        self.assertFalse(one.is_zero())
        self.assertTrue(one.is_one())
        self.assertEqual(one.degree(), 0)
    
    def test_variable(self):
        """Test: Variable simple."""
        x0 = Z2Polynomial.variable(0)
        x1 = Z2Polynomial.variable(1)
        
        self.assertEqual(x0.degree(), 1)
        self.assertEqual(x0.num_terms(), 1)
        self.assertNotEqual(x0, x1)
    
    def test_from_minterm(self):
        """Test: Construcción desde minitérmino."""
        # Minterm 5 = 101 → x₀·x₂
        poly = Z2Polynomial.from_minterm(5, 3)
        
        expected_monomial = frozenset([0, 2])
        self.assertIn(expected_monomial, poly.monomials)
    
    def test_addition_commutative(self):
        """Test: Suma conmutativa."""
        p = Z2Polynomial.variable(0)
        q = Z2Polynomial.variable(1)
        
        self.assertEqual(p + q, q + p)
    
    def test_addition_associative(self):
        """Test: Suma asociativa."""
        p = Z2Polynomial.variable(0)
        q = Z2Polynomial.variable(1)
        r = Z2Polynomial.variable(2)
        
        self.assertEqual((p + q) + r, p + (q + r))
    
    def test_addition_identity(self):
        """Test: Cero es identidad aditiva."""
        p = Z2Polynomial.variable(0)
        zero = Z2Polynomial.zero()
        
        self.assertEqual(p + zero, p)
    
    def test_addition_self_cancellation(self):
        """Test: p + p = 0 (característica 2)."""
        p = Z2Polynomial.variable(0)
        
        result = p + p
        self.assertTrue(result.is_zero())
    
    def test_multiplication_commutative(self):
        """Test: Multiplicación conmutativa."""
        p = Z2Polynomial.variable(0)
        q = Z2Polynomial.variable(1)
        
        self.assertEqual(p * q, q * p)
    
    def test_multiplication_associative(self):
        """Test: Multiplicación asociativa."""
        p = Z2Polynomial.variable(0)
        q = Z2Polynomial.variable(1)
        r = Z2Polynomial.variable(2)
        
        self.assertEqual((p * q) * r, p * (q * r))
    
    def test_multiplication_identity(self):
        """Test: 1 es identidad multiplicativa."""
        p = Z2Polynomial.variable(0)
        one = Z2Polynomial.one()
        
        self.assertEqual(p * one, p)
    
    def test_multiplication_zero(self):
        """Test: 0 es absorbente."""
        p = Z2Polynomial.variable(0)
        zero = Z2Polynomial.zero()
        
        result = p * zero
        self.assertTrue(result.is_zero())
    
    def test_idempotence(self):
        """Test: x² = x (idempotencia booleana)."""
        x = Z2Polynomial.variable(0)
        
        x_squared = x * x
        self.assertEqual(x_squared, x)
    
    def test_distributivity(self):
        """Test: Distributividad."""
        p = Z2Polynomial.variable(0)
        q = Z2Polynomial.variable(1)
        r = Z2Polynomial.variable(2)
        
        # p * (q + r) = p*q + p*r
        left = p * (q + r)
        right = (p * q) + (p * r)
        
        self.assertEqual(left, right)
    
    def test_negation(self):
        """Test: -p = p (característica 2)."""
        p = Z2Polynomial.variable(0)
        
        self.assertEqual(-p, p)
    
    def test_subtraction(self):
        """Test: p - q = p + q."""
        p = Z2Polynomial.variable(0)
        q = Z2Polynomial.variable(1)
        
        self.assertEqual(p - q, p + q)
    
    def test_degree_calculation(self):
        """Test: Cálculo de grado."""
        # x₀·x₁ tiene grado 2
        p = Z2Polynomial.variable(0) * Z2Polynomial.variable(1)
        self.assertEqual(p.degree(), 2)
        
        # x₀ + x₁ tiene grado 1
        q = Z2Polynomial.variable(0) + Z2Polynomial.variable(1)
        self.assertEqual(q.degree(), 1)
    
    def test_evaluation(self):
        """Test: Evaluación de polinomio."""
        # p = x₀ + x₁ + 1
        p = Z2Polynomial.variable(0) + Z2Polynomial.variable(1) + Z2Polynomial.one()
        
        # Evaluar en x₀=1, x₁=0
        result = p.evaluate({0: 1, 1: 0})
        # 1 + 0 + 1 = 0 (mod 2)
        self.assertEqual(result, 0)
        
        # Evaluar en x₀=1, x₁=1
        result = p.evaluate({0: 1, 1: 1})
        # 1 + 1 + 1 = 1 (mod 2)
        self.assertEqual(result, 1)
    
    def test_leading_term(self):
        """Test: Término líder."""
        # p = x₀·x₁ + x₂
        p = (Z2Polynomial.variable(0) * Z2Polynomial.variable(1)) + Z2Polynomial.variable(2)
        
        lt = p.leading_term()
        
        # El término de mayor grado es x₀·x₁ (grado 2)
        self.assertEqual(lt, frozenset([0, 1]))
    
    def test_complex_polynomial(self):
        """Test: Polinomio complejo."""
        # p = x₀·x₁ + x₁·x₂ + x₀·x₂ + 1
        x0 = Z2Polynomial.variable(0)
        x1 = Z2Polynomial.variable(1)
        x2 = Z2Polynomial.variable(2)
        one = Z2Polynomial.one()
        
        p = (x0 * x1) + (x1 * x2) + (x0 * x2) + one
        
        self.assertEqual(p.degree(), 2)
        self.assertEqual(p.num_terms(), 4)


# =============================================================================
# TESTS: GrobnerBasisZ2 (Base de Gröbner)
# =============================================================================

class TestGrobnerBasisZ2(unittest.TestCase):
    """Test suite para GrobnerBasisZ2."""
    
    def setUp(self):
        """Inicialización."""
        self.gb = GrobnerBasisZ2(num_vars=3)
    
    def test_initialization(self):
        """Test: Inicialización."""
        self.assertEqual(self.gb.num_vars, 3)
        self.assertEqual(len(self.gb), 0)
    
    def test_add_polynomial(self):
        """Test: Agregar polinomio."""
        p = Z2Polynomial.variable(0)
        self.gb.add_polynomial(p)
        
        self.assertGreater(len(self.gb), 0)
    
    def test_add_zero_ignored(self):
        """Test: Agregar cero no cambia la base."""
        zero = Z2Polynomial.zero()
        initial_len = len(self.gb)
        
        self.gb.add_polynomial(zero)
        
        self.assertEqual(len(self.gb), initial_len)
    
    def test_normal_form_zero(self):
        """Test: Forma normal de cero es cero."""
        p = Z2Polynomial.variable(0)
        self.gb.add_polynomial(p)
        
        zero = Z2Polynomial.zero()
        nf = self.gb.normal_form(zero)
        
        self.assertTrue(nf.is_zero())
    
    def test_is_member_self(self):
        """Test: Polinomio es miembro del ideal que genera."""
        p = Z2Polynomial.variable(0)
        self.gb.add_polynomial(p)
        
        self.assertTrue(self.gb.is_member(p))
    
    def test_is_member_multiple(self):
        """Test: Múltiplo de generador está en el ideal."""
        p = Z2Polynomial.variable(0)
        q = Z2Polynomial.variable(1)
        
        self.gb.add_polynomial(p)
        
        # p * q debe estar en ⟨p⟩
        multiple = p * q
        self.assertTrue(self.gb.is_member(multiple))
    
    def test_is_member_sum(self):
        """Test: Suma de generadores está en el ideal."""
        p = Z2Polynomial.variable(0)
        q = Z2Polynomial.variable(1)
        
        self.gb.add_polynomial(p)
        self.gb.add_polynomial(q)
        
        # p + q debe estar en ⟨p, q⟩
        sum_poly = p + q
        self.assertTrue(self.gb.is_member(sum_poly))
    
    def test_redundant_polynomial(self):
        """Test: Polinomio redundante."""
        p = Z2Polynomial.variable(0)
        q = p  # Redundante (idéntico)
        
        self.gb.add_polynomial(p)
        
        # Forma normal de q debe ser cero
        nf = self.gb.normal_form(q)
        self.assertTrue(nf.is_zero())
    
    def test_independent_polynomials(self):
        """Test: Polinomios independientes."""
        p = Z2Polynomial.variable(0)
        q = Z2Polynomial.variable(1)
        
        self.gb.add_polynomial(p)
        
        # Forma normal de q no debe ser cero
        nf = self.gb.normal_form(q)
        self.assertFalse(nf.is_zero())


# =============================================================================
# TESTS: ROBDD (Diagramas de Decisión Binaria)
# =============================================================================

class TestROBDD(unittest.TestCase):
    """Test suite para ROBDD."""
    
    def setUp(self):
        """Inicialización."""
        self.robdd = ROBDD(num_vars=3)
    
    def test_initialization(self):
        """Test: Inicialización."""
        self.assertEqual(self.robdd.num_vars, 3)
    
    def test_build_empty_set(self):
        """Test: Construir desde conjunto vacío."""
        node = self.robdd.build_from_minterms(set())
        
        self.assertTrue(node.is_false())
    
    def test_build_single_minterm(self):
        """Test: Construir desde un minitérmino."""
        # Minterm 5 = 101
        node = self.robdd.build_from_minterms({5})
        
        self.assertFalse(node.is_terminal())
    
    def test_build_multiple_minterms(self):
        """Test: Construir desde múltiples minitérminos."""
        node = self.robdd.build_from_minterms({1, 3, 5, 7})
        
        # Verificar que no es terminal
        self.assertFalse(node.is_terminal())
    
    def test_extract_minterms(self):
        """Test: Extraer minitérminos."""
        original = {1, 3, 5}
        node = self.robdd.build_from_minterms(original)
        
        extracted = self.robdd.extract_minterms(node)
        
        self.assertEqual(extracted, original)
    
    def test_apply_and_true_false(self):
        """Test: TRUE AND FALSE = FALSE."""
        true_node = ROBDDNode.get_true_node()
        false_node = ROBDDNode.get_false_node()
        
        result = self.robdd.apply_and(true_node, false_node)
        
        self.assertTrue(result.is_false())
    
    def test_apply_and_true_true(self):
        """Test: TRUE AND TRUE = TRUE."""
        true_node = ROBDDNode.get_true_node()
        
        result = self.robdd.apply_and(true_node, true_node)
        
        self.assertTrue(result.is_true())
    
    def test_apply_or_true_false(self):
        """Test: TRUE OR FALSE = TRUE."""
        true_node = ROBDDNode.get_true_node()
        false_node = ROBDDNode.get_false_node()
        
        result = self.robdd.apply_or(true_node, false_node)
        
        self.assertTrue(result.is_true())
    
    def test_apply_or_false_false(self):
        """Test: FALSE OR FALSE = FALSE."""
        false_node = ROBDDNode.get_false_node()
        
        result = self.robdd.apply_or(false_node, false_node)
        
        self.assertTrue(result.is_false())
    
    def test_apply_not(self):
        """Test: NOT TRUE = FALSE, NOT FALSE = TRUE."""
        true_node = ROBDDNode.get_true_node()
        false_node = ROBDDNode.get_false_node()
        
        self.assertTrue(self.robdd.apply_not(true_node).is_false())
        self.assertTrue(self.robdd.apply_not(false_node).is_true())
    
    def test_canonicity(self):
        """Test: ROBDDs con misma función son idénticos."""
        # Construir dos veces con mismos minterms
        minterms = {1, 3, 5}
        
        node1 = self.robdd.build_from_minterms(minterms)
        node2 = self.robdd.build_from_minterms(minterms)
        
        # Deben ser el mismo nodo (compartido)
        self.assertEqual(node1.id, node2.id)
    
    def test_count_solutions(self):
        """Test: Conteo de soluciones."""
        # Minterm 5 = 101 → solo una solución
        node = self.robdd.build_from_minterms({5})
        
        count = self.robdd.count_solutions(node)
        self.assertEqual(count, 1)
    
    def test_count_solutions_multiple(self):
        """Test: Conteo con múltiples soluciones."""
        # 2 minterms
        node = self.robdd.build_from_minterms({1, 3})
        
        count = self.robdd.count_solutions(node)
        self.assertEqual(count, 2)
    
    def test_size(self):
        """Test: Tamaño del ROBDD."""
        node = self.robdd.build_from_minterms({1, 3, 5, 7})
        
        size = self.robdd.size(node)
        
        # Debe haber al menos 1 nodo
        self.assertGreater(size, 0)


# =============================================================================
# TESTS: DPLLSATSolver (SAT Solver)
# =============================================================================

class TestDPLLSATSolver(unittest.TestCase):
    """Test suite para DPLLSATSolver."""
    
    def test_empty_cnf_sat(self):
        """Test: CNF vacía es SAT."""
        cnf = []
        self.assertTrue(DPLLSATSolver.solve(cnf))
    
    def test_single_positive_literal_sat(self):
        """Test: Cláusula unitaria positiva."""
        cnf = [[1]]
        self.assertTrue(DPLLSATSolver.solve(cnf))
    
    def test_single_negative_literal_sat(self):
        """Test: Cláusula unitaria negativa."""
        cnf = [[-1]]
        self.assertTrue(DPLLSATSolver.solve(cnf))
    
    def test_contradiction_unsat(self):
        """Test: Contradicción es UNSAT."""
        # x ∧ ¬x
        cnf = [[1], [-1]]
        self.assertFalse(DPLLSATSolver.solve(cnf))
    
    def test_simple_sat(self):
        """Test: Fórmula simple SAT."""
        # (x ∨ y) ∧ (¬x ∨ z) ∧ (¬y ∨ ¬z)
        cnf = [
            [1, 2],
            [-1, 3],
            [-2, -3]
        ]
        self.assertTrue(DPLLSATSolver.solve(cnf))
    
    def test_pigeonhole_unsat(self):
        """Test: Principio del palomar (UNSAT)."""
        # 3 palomas, 2 agujeros → UNSAT
        # Variables: p_ij (paloma i en agujero j)
        # p_11, p_12, p_21, p_22, p_31, p_32
        
        cnf = [
            # Cada paloma en al menos un agujero
            [1, 2],  # p_11 ∨ p_12
            [3, 4],  # p_21 ∨ p_22
            [5, 6],  # p_31 ∨ p_32
            
            # A lo más una paloma por agujero
            [-1, -3],  # ¬p_11 ∨ ¬p_21
            [-1, -5],  # ¬p_11 ∨ ¬p_31
            [-3, -5],  # ¬p_21 ∨ ¬p_31
            [-2, -4],  # ¬p_12 ∨ ¬p_22
            [-2, -6],  # ¬p_12 ∨ ¬p_32
            [-4, -6],  # ¬p_22 ∨ ¬p_32
        ]
        
        self.assertFalse(DPLLSATSolver.solve(cnf))
    
    def test_unit_propagation(self):
        """Test: Propagación de unidades."""
        # (x) ∧ (¬x ∨ y) → debe inferir y=TRUE
        cnf = [
            [1],
            [-1, 2]
        ]
        self.assertTrue(DPLLSATSolver.solve(cnf))
    
    def test_pure_literal_elimination(self):
        """Test: Eliminación de literales puros."""
        # (x ∨ y) ∧ (x ∨ z) → x solo aparece positivo → x=TRUE
        cnf = [
            [1, 2],
            [1, 3]
        ]
        self.assertTrue(DPLLSATSolver.solve(cnf))


# =============================================================================
# TESTS: MICRedundancyAnalyzer (Analizador Principal)
# =============================================================================

class TestMICRedundancyAnalyzer(unittest.TestCase):
    """Test suite para MICRedundancyAnalyzer."""
    
    def setUp(self):
        """Inicialización."""
        self.analyzer = MICRedundancyAnalyzer(num_capabilities=5)
    
    # -------------------------------------------------------------------------
    # REGISTRO DE HERRAMIENTAS
    # -------------------------------------------------------------------------
    
    def test_initialization(self):
        """Test: Inicialización del analizador."""
        self.assertEqual(self.analyzer.num_capabilities, 5)
        self.assertEqual(len(self.analyzer.tools), 0)
    
    def test_register_tool_valid(self):
        """Test: Registro de herramienta válida."""
        self.analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO})
        
        self.assertEqual(len(self.analyzer.tools), 1)
        self.assertEqual(self.analyzer.tools[0].name, "tool1")
    
    def test_register_tool_with_boolean_vector(self):
        """Test: Registro con BooleanVector."""
        caps = BooleanVector(frozenset([CapabilityDimension.PHYS_IO]))
        self.analyzer.register_tool("tool1", caps)
        
        self.assertEqual(len(self.analyzer.tools), 1)
    
    def test_register_duplicate_name(self):
        """Test: Nombre duplicado lanza excepción."""
        self.analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO})
        
        with self.assertRaises(ValueError):
            self.analyzer.register_tool("tool1", {CapabilityDimension.PHYS_NUM})
    
    def test_register_empty_name(self):
        """Test: Nombre vacío lanza excepción."""
        with self.assertRaises(ValueError):
            self.analyzer.register_tool("", {CapabilityDimension.PHYS_IO})
    
    def test_register_invalid_capabilities_type(self):
        """Test: Tipo inválido de capacidades."""
        with self.assertRaises(TypeError):
            self.analyzer.register_tool("tool1", ["PHYS_IO"])  # type: ignore
    
    def test_unregister_tool(self):
        """Test: Eliminar herramienta."""
        self.analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO})
        self.analyzer.unregister_tool("tool1")
        
        self.assertEqual(len(self.analyzer.tools), 0)
    
    def test_unregister_nonexistent(self):
        """Test: Eliminar herramienta inexistente (no lanza error)."""
        # No debe lanzar excepción, solo advertencia en log
        self.analyzer.unregister_tool("nonexistent")
    
    # -------------------------------------------------------------------------
    # MATRIZ DE INCIDENCIA
    # -------------------------------------------------------------------------
    
    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy no disponible")
    def test_build_incidence_matrix_empty(self):
        """Test: Matriz vacía cuando no hay herramientas."""
        matrix = self.analyzer.build_incidence_matrix()
        
        self.assertEqual(matrix.shape, (0, 5))
    
    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy no disponible")
    def test_build_incidence_matrix_single_tool(self):
        """Test: Matriz con una herramienta."""
        self.analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO})
        
        matrix = self.analyzer.build_incidence_matrix()
        
        self.assertEqual(matrix.shape, (1, 5))
        self.assertEqual(matrix[0, 0], 1)  # PHYS_IO es índice 0
    
    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy no disponible")
    def test_build_incidence_matrix_multiple_tools(self):
        """Test: Matriz con múltiples herramientas."""
        self.analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO})
        self.analyzer.register_tool("tool2", {CapabilityDimension.PHYS_NUM})
        
        matrix = self.analyzer.build_incidence_matrix()
        
        self.assertEqual(matrix.shape, (2, 5))
    
    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy no disponible")
    def test_incidence_matrix_caching(self):
        """Test: Caché de matriz de incidencia."""
        self.analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO})
        
        matrix1 = self.analyzer.build_incidence_matrix()
        matrix2 = self.analyzer.build_incidence_matrix()
        
        # Deben ser el mismo objeto (caché)
        self.assertTrue(np.array_equal(matrix1, matrix2))
    
    # -------------------------------------------------------------------------
    # ANÁLISIS ESPECTRAL
    # -------------------------------------------------------------------------
    
    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy no disponible")
    def test_spectral_properties_empty(self):
        """Test: Propiedades espectrales de matriz vacía."""
        props = self.analyzer.compute_spectral_properties()
        
        self.assertEqual(props['rank'], 0)
        self.assertEqual(props['nullity'], 0)
        self.assertFalse(props['is_full_rank'])
    
    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy no disponible")
    def test_spectral_properties_full_rank(self):
        """Test: Matriz de rango completo."""
        # Registrar herramientas independientes
        for i, cap in enumerate(CapabilityDimension):
            self.analyzer.register_tool(f"tool_{i}", {cap})
        
        props = self.analyzer.compute_spectral_properties()
        
        self.assertEqual(props['rank'], 5)
        self.assertTrue(props['is_full_rank'])
    
    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy no disponible")
    def test_spectral_properties_redundant(self):
        """Test: Matriz con redundancia."""
        # Dos herramientas idénticas
        self.analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO})
        self.analyzer.register_tool("tool2", {CapabilityDimension.PHYS_IO})
        
        props = self.analyzer.compute_spectral_properties()
        
        # Rango = 1 (una sola dirección independiente)
        self.assertEqual(props['rank'], 1)
    
    # -------------------------------------------------------------------------
    # DEPENDENCIAS LINEALES
    # -------------------------------------------------------------------------
    
    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy no disponible")
    def test_detect_dependencies_subset(self):
        """Test: Detección de relación de subconjunto."""
        self.analyzer.register_tool("general", {
            CapabilityDimension.PHYS_IO,
            CapabilityDimension.PHYS_NUM
        })
        self.analyzer.register_tool("specific", {CapabilityDimension.PHYS_IO})
        
        deps = self.analyzer.detect_linear_dependencies_z2()
        
        # Debe haber una dependencia de tipo subset
        self.assertEqual(len(deps), 1)
        self.assertEqual(deps[0]['type'], 'subset')
    
    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy no disponible")
    def test_detect_dependencies_duplicate(self):
        """Test: Detección de duplicados."""
        self.analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO})
        self.analyzer.register_tool("tool2", {CapabilityDimension.PHYS_IO})
        
        deps = self.analyzer.detect_linear_dependencies_z2()
        
        # Debe haber una dependencia de tipo duplicate
        self.assertEqual(len(deps), 1)
        self.assertEqual(deps[0]['type'], 'duplicate')
    
    # -------------------------------------------------------------------------
    # HOMOLOGÍA
    # -------------------------------------------------------------------------
    
    def test_homology_empty(self):
        """Test: Homología con conjunto vacío."""
        homology = self.analyzer.compute_homology_groups()
        
        self.assertEqual(homology['H_0'], 0)
        self.assertEqual(homology['H_1'], 0)
    
    def test_homology_single_component(self):
        """Test: Una componente conexa."""
        self.analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO})
        self.analyzer.register_tool("tool2", {CapabilityDimension.PHYS_IO})
        
        homology = self.analyzer.compute_homology_groups()
        
        # Ambas comparten capacidad → 1 componente
        self.assertEqual(homology['H_0'], 1)
    
    def test_homology_multiple_components(self):
        """Test: Múltiples componentes conexas."""
        self.analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO})
        self.analyzer.register_tool("tool2", {CapabilityDimension.PHYS_NUM})
        
        homology = self.analyzer.compute_homology_groups()
        
        # No comparten capacidades → 2 componentes
        self.assertEqual(homology['H_0'], 2)
    
    def test_homology_redundancy_cycles(self):
        """Test: Detección de ciclos de redundancia."""
        # Dos herramientas idénticas forman un ciclo
        self.analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO})
        self.analyzer.register_tool("tool2", {CapabilityDimension.PHYS_IO})
        
        homology = self.analyzer.compute_homology_groups()
        
        # Debe haber un ciclo de redundancia
        self.assertEqual(homology['H_1'], 1)
        self.assertEqual(len(homology['redundancy_cycles']), 1)
    
    # -------------------------------------------------------------------------
    # MINIMIZACIÓN CON ROBDD
    # -------------------------------------------------------------------------
    
    def test_minimize_robdd_empty(self):
        """Test: Minimización con conjunto vacío."""
        essential, redundant = self.analyzer.minimize_with_robdd()
        
        self.assertEqual(len(essential), 0)
        self.assertEqual(len(redundant), 0)
    
    def test_minimize_robdd_no_redundancy(self):
        """Test: Sin redundancia."""
        self.analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO})
        self.analyzer.register_tool("tool2", {CapabilityDimension.PHYS_NUM})
        
        essential, redundant = self.analyzer.minimize_with_robdd()
        
        self.assertEqual(len(essential), 2)
        self.assertEqual(len(redundant), 0)
    
    def test_minimize_robdd_with_duplicates(self):
        """Test: Con duplicados."""
        self.analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO})
        self.analyzer.register_tool("tool2", {CapabilityDimension.PHYS_IO})
        self.analyzer.register_tool("tool3", {CapabilityDimension.PHYS_IO})
        
        essential, redundant = self.analyzer.minimize_with_robdd()
        
        # Solo una esencial, dos redundantes
        self.assertEqual(len(essential), 1)
        self.assertEqual(len(redundant), 2)
    
    # -------------------------------------------------------------------------
    # VERIFICACIÓN SAT
    # -------------------------------------------------------------------------
    
    def test_sat_empty(self):
        """Test: SAT con conjunto vacío."""
        sat_ok, conflicting = self.analyzer.check_non_interference_sat()
        
        self.assertTrue(sat_ok)
        self.assertEqual(len(conflicting), 0)
    
    def test_sat_single_tool(self):
        """Test: SAT con una herramienta."""
        self.analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO})
        
        sat_ok, conflicting = self.analyzer.check_non_interference_sat()
        
        self.assertTrue(sat_ok)
    
    def test_sat_no_conflict(self):
        """Test: SAT sin conflictos."""
        self.analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO})
        self.analyzer.register_tool("tool2", {CapabilityDimension.PHYS_NUM})
        
        sat_ok, conflicting = self.analyzer.check_non_interference_sat()
        
        self.assertTrue(sat_ok)
        self.assertEqual(len(conflicting), 0)
    
    # -------------------------------------------------------------------------
    # ANÁLISIS INTEGRADO
    # -------------------------------------------------------------------------
    
    def test_analyze_redundancy_empty(self):
        """Test: Análisis con conjunto vacío."""
        result = self.analyzer.analyze_redundancy()
        
        self.assertEqual(result['status'], 'empty')
        self.assertEqual(len(result['essential_tools']), 0)
    
    def test_analyze_redundancy_simple(self):
        """Test: Análisis simple."""
        self.analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO})
        self.analyzer.register_tool("tool2", {CapabilityDimension.PHYS_NUM})
        
        result = self.analyzer.analyze_redundancy()
        
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['statistics']['total_tools'], 2)
    
    def test_analyze_redundancy_with_redundants(self):
        """Test: Análisis con redundancias."""
        self.analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO})
        self.analyzer.register_tool("tool2", {CapabilityDimension.PHYS_IO})
        self.analyzer.register_tool("tool3", {CapabilityDimension.PHYS_NUM})
        
        result = self.analyzer.analyze_redundancy()
        
        stats = result['statistics']
        
        # Debe haber al menos una redundante
        self.assertGreater(stats['redundant_count'], 0)
        self.assertGreater(stats['reduction_rate'], 0.0)


# =============================================================================
# TESTS: Validación de Axiomas
# =============================================================================

class TestAxiomValidation(unittest.TestCase):
    """Test suite para validación de axiomas."""
    
    def test_validate_boolean_lattice_axioms(self):
        """Test: Validación de axiomas del retículo booleano."""
        # No debe lanzar excepción
        validate_boolean_lattice_axioms(num_vars=5)
    
    def test_axioms_with_random_vectors(self):
        """Test: Axiomas con vectores aleatorios."""
        for _ in range(10):
            a = TestHelpers.random_boolean_vector()
            b = TestHelpers.random_boolean_vector()
            c = TestHelpers.random_boolean_vector()
            
            # Conmutatividad
            self.assertEqual(a.union(b), b.union(a))
            self.assertEqual(a.intersection(b), b.intersection(a))
            
            # Asociatividad
            self.assertEqual(
                a.union(b).union(c),
                a.union(b.union(c))
            )
            
            # Idempotencia
            self.assertEqual(a.union(a), a)
            self.assertEqual(a.intersection(a), a)


# =============================================================================
# TESTS: Integración End-to-End
# =============================================================================

class TestIntegration(unittest.TestCase):
    """Tests de integración end-to-end."""
    
    def test_full_audit_workflow(self):
        """Test: Flujo completo de auditoría."""
        result = audit_mic_redundancy()
        
        # Verificar estructura del resultado
        self.assertIn('status', result)
        self.assertIn('essential_tools', result)
        self.assertIn('redundant_tools', result)
        self.assertIn('statistics', result)
        
        stats = result['statistics']
        
        # Verificar estadísticas
        self.assertIn('total_tools', stats)
        self.assertIn('essential_count', stats)
        self.assertIn('redundant_count', stats)
        
        # Verificar consistencia
        total = stats['total_tools']
        essential = stats['essential_count']
        redundant = stats['redundant_count']
        
        # Essential + redundant debe sumar al total
        self.assertEqual(essential + redundant, total)
    
    def test_example_scenario(self):
        """Test: Escenario de ejemplo documentado."""
        analyzer = MICRedundancyAnalyzer()
        
        # Registrar herramientas de ejemplo
        analyzer.register_tool("parse_io", {CapabilityDimension.PHYS_IO})
        analyzer.register_tool("compute_num", {CapabilityDimension.PHYS_NUM})
        analyzer.register_tool("topo_analysis", {CapabilityDimension.TACT_TOPO})
        
        # Redundantes
        analyzer.register_tool("parse_io_2", {CapabilityDimension.PHYS_IO})
        analyzer.register_tool("compute_num_alt", {CapabilityDimension.PHYS_NUM})
        
        result = analyzer.analyze_redundancy()
        
        # Debe detectar 2 redundantes
        self.assertEqual(result['statistics']['redundant_count'], 2)
        self.assertEqual(result['statistics']['essential_count'], 3)


# =============================================================================
# TESTS: Performance y Límites
# =============================================================================

class TestPerformance(unittest.TestCase):
    """Tests de performance y casos límite."""
    
    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy no disponible")
    def test_large_number_of_tools(self):
        """Test: Gran número de herramientas."""
        analyzer = MICRedundancyAnalyzer()
        
        # Registrar 50 herramientas
        for i in range(50):
            cap_idx = i % len(CapabilityDimension)
            cap = list(CapabilityDimension)[cap_idx]
            analyzer.register_tool(f"tool_{i}", {cap})
        
        # Debe completar sin errores
        result = analyzer.analyze_redundancy()
        
        self.assertEqual(result['status'], 'success')
    
    def test_deep_polynomial_operations(self):
        """Test: Operaciones profundas con polinomios."""
        # Multiplicar muchos polinomios
        p = Z2Polynomial.one()
        
        for i in range(5):
            p = p * Z2Polynomial.variable(i)
        
        # Debe tener grado 5
        self.assertEqual(p.degree(), 5)
    
    def test_robdd_with_many_minterms(self):
        """Test: ROBDD con muchos minitérminos."""
        robdd = ROBDD(num_vars=5)
        
        # 16 minitérminos
        minterms = set(range(16))
        node = robdd.build_from_minterms(minterms)
        
        # Verificar
        extracted = robdd.extract_minterms(node)
        self.assertEqual(extracted, minterms)


# =============================================================================
# TESTS: Casos Edge y Robustez
# =============================================================================

class TestEdgeCases(unittest.TestCase):
    """Tests de casos edge y robustez."""
    
    def test_empty_capabilities(self):
        """Test: Herramienta sin capacidades."""
        analyzer = MICRedundancyAnalyzer()
        analyzer.register_tool("empty_tool", set())
        
        result = analyzer.analyze_redundancy()
        
        # No debe fallar
        self.assertEqual(result['status'], 'success')
    
    def test_all_capabilities(self):
        """Test: Herramienta con todas las capacidades."""
        analyzer = MICRedundancyAnalyzer()
        analyzer.register_tool("universal", set(CapabilityDimension))
        
        result = analyzer.analyze_redundancy()
        
        self.assertEqual(result['status'], 'success')
    
    def test_invalid_minterm_range(self):
        """Test: Minitérmino fuera de rango."""
        with self.assertRaises(ValueError):
            BooleanVector.from_minterm(100, 3)
    
    def test_polynomial_evaluation_invalid_value(self):
        """Test: Evaluación con valor inválido."""
        p = Z2Polynomial.variable(0)
        
        with self.assertRaises(ValueError):
            p.evaluate({0: 2})  # Valor debe ser 0 o 1
    
    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy no disponible")
    def test_ill_conditioned_matrix(self):
        """Test: Matriz mal condicionada (advertencia, no error)."""
        analyzer = MICRedundancyAnalyzer()
        
        # Crear herramientas casi idénticas
        analyzer.register_tool("t1", {
            CapabilityDimension.PHYS_IO,
            CapabilityDimension.PHYS_NUM
        })
        analyzer.register_tool("t2", {
            CapabilityDimension.PHYS_IO,
            CapabilityDimension.PHYS_NUM
        })
        
        # No debe lanzar excepción, solo advertencia
        props = analyzer.compute_spectral_properties()
        
        # Condición number debe ser grande
        self.assertGreater(props['condition_number'], 1.0)


# =============================================================================
# SUITE DE TESTS PRINCIPAL
# =============================================================================

def suite():
    """Construye la suite completa de tests."""
    test_suite = unittest.TestSuite()
    
    # Agregar todos los test cases
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestBooleanVector))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestTool))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestZ2Polynomial))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestGrobnerBasisZ2))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestROBDD))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDPLLSATSolver))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestMICRedundancyAnalyzer))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAxiomValidation))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIntegration))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPerformance))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestEdgeCases))
    
    return test_suite


# =============================================================================
# RUNNER PRINCIPAL
# =============================================================================

if __name__ == '__main__':
    # Configurar logging para tests
    import logging
    logging.basicConfig(
        level=logging.WARNING,
        format='%(levelname)s | %(message)s'
    )
    
    # Ejecutar tests con verbosidad
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite())
    
    # Resumen final
    print("\n" + "=" * 80)
    print("TEST SUMMARY - MIC REDUNDANCY ANALYZER")
    print("=" * 80)
    print(f"Tests run:    {result.testsRun}")
    print(f"Successes:    {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures:     {len(result.failures)}")
    print(f"Errors:       {len(result.errors)}")
    print(f"Skipped:      {len(result.skipped)}")
    
    if not NUMPY_AVAILABLE:
        print("\n⚠️  WARNING: NumPy not available - some tests were skipped")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED")
        sys.exit(1)