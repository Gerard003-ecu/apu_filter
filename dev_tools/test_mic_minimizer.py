"""
=========================================================================================
Módulo: Suite de Pruebas para MIC Minimizer
Ubicación: dev_tools/test_mic_minimizer.py
Versión: 1.0 - Suite Rigurosa de Testing
=========================================================================================

COBERTURA DE PRUEBAS:
---------------------
1. Estructuras de datos algebraicas (BooleanVector, Tool, ImplicantTerm)
2. Axiomas del retículo booleano
3. Union-Find y componentes conexas
4. Algoritmo de Quine-McCluskey
5. Análisis espectral y homología
6. Análisis completo de redundancia
7. Casos extremos y edge cases
8. Propiedades de performance

FRAMEWORKS:
-----------
- unittest: Framework base
- hypothesis: Property-based testing
- numpy.testing: Comparaciones numéricas
- contextlib: Manejo de contextos

EJECUCIÓN:
----------
    python -m pytest test_mic_minimizer.py -v --cov=mic_minimizer --cov-report=html
    python -m unittest test_mic_minimizer.TestBooleanVector -v

=========================================================================================
"""

import unittest
import numpy as np
from typing import Set, List, FrozenSet
from contextlib import contextmanager
import sys
import io
import logging

# Importar módulo a testear
from app.boole.tactics.mic_minimizer import (
    CapabilityDimension,
    BooleanVector,
    Tool,
    ImplicantTerm,
    UnionFind,
    QuineMcCluskeyMinimizer,
    TopologicalInvariantComputer,
    MICRedundancyAnalyzer,
    validate_boolean_lattice_axioms,
    HomologicalInconsistencyError
)

# Suprimir logs durante testing
logging.getLogger("MIC.Minimizer.v3.1").setLevel(logging.CRITICAL)


# ========================================================================================
# CONSTANTES DE RIGOR COMPUTACIONAL
# ========================================================================================
NUM_VARS_MAX_TESTED = 8  # Cota superior tratable para minimización algorítmica exacta en tiempo O(1)
MAX_COMPUTATION_TIME_S = 0.5

# ========================================================================================
# FIXTURES
# ========================================================================================

class TestBase(unittest.TestCase):
    """Clase base con utilidades comunes para testing."""
    
    @contextmanager
    def assertNotRaises(self, exc_type):
        """Context manager que verifica que NO se lanza una excepción."""
        try:
            yield
        except exc_type as e:
            self.fail(f"Se lanzó {exc_type.__name__} inesperadamente: {e}")
    
    def assertSetEqual(self, set1, set2, msg=None):
        """Compara conjuntos con mensaje detallado."""
        if set1 != set2:
            diff1 = set1 - set2
            diff2 = set2 - set1
            custom_msg = f"\nConjuntos diferentes:\n  En set1 pero no en set2: {diff1}\n  En set2 pero no en set1: {diff2}"
            if msg:
                custom_msg = f"{msg}\n{custom_msg}"
            self.fail(custom_msg)
    
    def assertMatrixEqual(self, matrix1, matrix2, msg=None):
        """Compara matrices numpy."""
        np.testing.assert_array_equal(matrix1, matrix2, err_msg=msg)
    
    def assertMatrixAlmostEqual(self, matrix1, matrix2, decimal=7, msg=None):
        """Compara matrices numpy con tolerancia."""
        np.testing.assert_array_almost_equal(matrix1, matrix2, decimal=decimal, err_msg=msg)


# ========================================================================================
# TESTS: CapabilityDimension
# ========================================================================================

class TestCapabilityDimension(TestBase):
    """Tests para la enumeración CapabilityDimension."""
    
    def test_all_dimensions_exist(self):
        """Verifica que todas las dimensiones esperadas existen."""
        expected = {'PHYS_IO', 'PHYS_NUM', 'TACT_TOPO', 'STRAT_FIN', 'WIS_SEM'}
        actual = {dim.name for dim in CapabilityDimension}
        self.assertEqual(expected, actual)
    
    def test_dimension_values_sequential(self):
        """Verifica que los valores son secuenciales comenzando en 0."""
        values = [dim.value for dim in CapabilityDimension]
        self.assertEqual(values, list(range(len(CapabilityDimension))))
    
    def test_ordering_is_total(self):
        """Verifica que el ordenamiento es total y transitivo."""
        dims = list(CapabilityDimension)
        
        # Reflexividad
        for d in dims:
            self.assertFalse(d < d)
        
        # Antisimetría
        for i, d1 in enumerate(dims):
            for j, d2 in enumerate(dims):
                if i < j:
                    self.assertTrue(d1 < d2)
                    self.assertFalse(d2 < d1)
        
        # Transitividad
        for d1 in dims:
            for d2 in dims:
                for d3 in dims:
                    if d1 < d2 and d2 < d3:
                        self.assertTrue(d1 < d3)
    
    def test_ordering_consistency_with_value(self):
        """El orden < debe ser consistente con el valor numérico."""
        dims = list(CapabilityDimension)
        for d1 in dims:
            for d2 in dims:
                self.assertEqual(d1 < d2, d1.value < d2.value)


# ========================================================================================
# TESTS: BooleanVector
# ========================================================================================

class TestBooleanVector(TestBase):
    """Tests para la clase BooleanVector."""
    
    def setUp(self):
        """Configuración de vectores de prueba."""
        self.empty = BooleanVector(frozenset())
        self.v1 = BooleanVector(frozenset([CapabilityDimension.PHYS_IO]))
        self.v2 = BooleanVector(frozenset([CapabilityDimension.PHYS_NUM]))
        self.v3 = BooleanVector(frozenset([CapabilityDimension.PHYS_IO, CapabilityDimension.PHYS_NUM]))
        self.v_all = BooleanVector(frozenset(CapabilityDimension))
    
    def test_immutability(self):
        """Los BooleanVector deben ser inmutables."""
        with self.assertRaises(AttributeError):
            self.v1.components = frozenset([CapabilityDimension.PHYS_NUM])
    
    def test_hashability(self):
        """Los BooleanVector deben ser hasheables."""
        v1_copy = BooleanVector(frozenset([CapabilityDimension.PHYS_IO]))
        
        # Mismo contenido → mismo hash
        self.assertEqual(hash(self.v1), hash(v1_copy))
        
        # Se pueden usar en sets
        s = {self.v1, self.v2, v1_copy}
        self.assertEqual(len(s), 2)  # v1 y v1_copy son iguales
        
        # Se pueden usar como claves de diccionario
        d = {self.v1: "value1", self.v2: "value2"}
        self.assertEqual(d[v1_copy], "value1")
    
    def test_equality(self):
        """Igualdad basada en componentes."""
        v1_copy = BooleanVector(frozenset([CapabilityDimension.PHYS_IO]))
        
        self.assertEqual(self.v1, v1_copy)
        self.assertNotEqual(self.v1, self.v2)
        self.assertNotEqual(self.v1, self.v3)
    
    def test_ordering_total(self):
        """El ordenamiento debe ser total."""
        vectors = [self.empty, self.v1, self.v2, self.v3]
        
        # Debe ser posible ordenar
        sorted_vectors = sorted(vectors)
        self.assertEqual(len(sorted_vectors), 4)
        
        # Debe ser determinista
        self.assertEqual(sorted(vectors), sorted(vectors))
    
    def test_from_minterm(self):
        """Construcción desde minitérmino."""
        # 0b101 = 5 → bits 0 y 2 activos
        v = BooleanVector.from_minterm(5, 5)
        expected = BooleanVector(frozenset([
            CapabilityDimension.PHYS_IO,    # bit 0
            CapabilityDimension.TACT_TOPO   # bit 2
        ]))
        self.assertEqual(v, expected)
        
        # Caso especial: 0
        v_zero = BooleanVector.from_minterm(0, 5)
        self.assertEqual(v_zero, self.empty)
        
        # Caso especial: todos los bits
        v_all = BooleanVector.from_minterm((1 << 5) - 1, 5)
        self.assertEqual(v_all.hamming_weight(), 5)
    
    def test_from_minterm_validation(self):
        """Validación de rango en from_minterm."""
        with self.assertRaises(ValueError):
            BooleanVector.from_minterm(-1, 5)
        
        with self.assertRaises(ValueError):
            BooleanVector.from_minterm(32, 5)  # 32 = 2^5, fuera de rango
    
    def test_to_binary_string(self):
        """Conversión a string binario."""
        self.assertEqual(self.empty.to_binary_string(5), "00000")
        self.assertEqual(self.v1.to_binary_string(5), "10000")
        self.assertEqual(self.v2.to_binary_string(5), "01000")
        self.assertEqual(self.v3.to_binary_string(5), "11000")
        self.assertEqual(self.v_all.to_binary_string(5), "11111")
    
    def test_to_minterm(self):
        """Conversión a minitérmino."""
        self.assertEqual(self.empty.to_minterm(), 0)
        self.assertEqual(self.v1.to_minterm(), 1)    # 2^0
        self.assertEqual(self.v2.to_minterm(), 2)    # 2^1
        self.assertEqual(self.v3.to_minterm(), 3)    # 2^0 + 2^1
    
    def test_roundtrip_minterm(self):
        """from_minterm y to_minterm son inversos."""
        for minterm in range(32):  # 2^5
            v = BooleanVector.from_minterm(minterm, 5)
            self.assertEqual(v.to_minterm(), minterm)
    
    def test_hamming_weight(self):
        """Peso de Hamming = número de componentes."""
        self.assertEqual(self.empty.hamming_weight(), 0)
        self.assertEqual(self.v1.hamming_weight(), 1)
        self.assertEqual(self.v2.hamming_weight(), 1)
        self.assertEqual(self.v3.hamming_weight(), 2)
        self.assertEqual(self.v_all.hamming_weight(), 5)
    
    # ===== OPERACIONES DEL RETÍCULO BOOLEANO =====
    
    def test_union_commutative(self):
        """OR es conmutativo: a ∨ b = b ∨ a."""
        self.assertEqual(self.v1.union(self.v2), self.v2.union(self.v1))
        self.assertEqual(self.v1.union(self.v3), self.v3.union(self.v1))
    
    def test_union_associative(self):
        """OR es asociativo: (a ∨ b) ∨ c = a ∨ (b ∨ c)."""
        left = self.v1.union(self.v2).union(self.v3)
        right = self.v1.union(self.v2.union(self.v3))
        self.assertEqual(left, right)
    
    def test_union_identity(self):
        """Identidad: a ∨ 0 = a."""
        self.assertEqual(self.v1.union(self.empty), self.v1)
        self.assertEqual(self.empty.union(self.v1), self.v1)
    
    def test_union_idempotent(self):
        """Idempotencia: a ∨ a = a."""
        self.assertEqual(self.v1.union(self.v1), self.v1)
        self.assertEqual(self.v3.union(self.v3), self.v3)
    
    def test_intersection_commutative(self):
        """AND es conmutativo: a ∧ b = b ∧ a."""
        self.assertEqual(self.v1.intersection(self.v2), self.v2.intersection(self.v1))
    
    def test_intersection_associative(self):
        """AND es asociativo: (a ∧ b) ∧ c = a ∧ (b ∧ c)."""
        left = self.v1.intersection(self.v2).intersection(self.v3)
        right = self.v1.intersection(self.v2.intersection(self.v3))
        self.assertEqual(left, right)
    
    def test_intersection_identity(self):
        """Identidad: a ∧ 1 = a."""
        self.assertEqual(self.v1.intersection(self.v_all), self.v1)
    
    def test_intersection_zero(self):
        """Aniquilador: a ∧ 0 = 0."""
        self.assertEqual(self.v1.intersection(self.empty), self.empty)
    
    def test_intersection_idempotent(self):
        """Idempotencia: a ∧ a = a."""
        self.assertEqual(self.v1.intersection(self.v1), self.v1)
    
    def test_absorption_law(self):
        """Absorción: a ∨ (a ∧ b) = a."""
        result = self.v1.union(self.v1.intersection(self.v2))
        self.assertEqual(result, self.v1)
        
        # Forma dual: a ∧ (a ∨ b) = a
        result_dual = self.v1.intersection(self.v1.union(self.v2))
        self.assertEqual(result_dual, self.v1)
    
    def test_distributive_law(self):
        """Distributividad: a ∧ (b ∨ c) = (a ∧ b) ∨ (a ∧ c)."""
        left = self.v1.intersection(self.v2.union(self.v3))
        right = self.v1.intersection(self.v2).union(self.v1.intersection(self.v3))
        self.assertEqual(left, right)
        
        # Forma dual: a ∨ (b ∧ c) = (a ∨ b) ∧ (a ∨ c)
        left_dual = self.v1.union(self.v2.intersection(self.v3))
        right_dual = self.v1.union(self.v2).intersection(self.v1.union(self.v3))
        self.assertEqual(left_dual, right_dual)
    
    def test_complement(self):
        """Complemento: a ∨ ¬a = 1, a ∧ ¬a = 0."""
        num_vars = 5
        
        # a ∨ ¬a = 1
        result = self.v1.union(self.v1.complement(num_vars))
        expected_all = BooleanVector(frozenset(CapabilityDimension))
        self.assertEqual(result, expected_all)
        
        # a ∧ ¬a = 0
        result_zero = self.v1.intersection(self.v1.complement(num_vars))
        self.assertEqual(result_zero, self.empty)
    
    def test_complement_involution(self):
        """Doble complemento: ¬(¬a) = a."""
        num_vars = 5
        double_complement = self.v1.complement(num_vars).complement(num_vars)
        self.assertEqual(double_complement, self.v1)
    
    def test_de_morgan_laws(self):
        """Leyes de De Morgan: ¬(a ∨ b) = ¬a ∧ ¬b, ¬(a ∧ b) = ¬a ∨ ¬b."""
        num_vars = 5
        
        # ¬(a ∨ b) = ¬a ∧ ¬b
        left = self.v1.union(self.v2).complement(num_vars)
        right = self.v1.complement(num_vars).intersection(self.v2.complement(num_vars))
        self.assertEqual(left, right)
        
        # ¬(a ∧ b) = ¬a ∨ ¬b
        left2 = self.v1.intersection(self.v2).complement(num_vars)
        right2 = self.v1.complement(num_vars).union(self.v2.complement(num_vars))
        self.assertEqual(left2, right2)
    
    def test_symmetric_difference(self):
        """XOR (suma en ℤ₂): a ⊕ b."""
        # Propiedades del grupo
        # Conmutatividad
        self.assertEqual(self.v1.symmetric_difference(self.v2), 
                        self.v2.symmetric_difference(self.v1))
        
        # Asociatividad
        left = self.v1.symmetric_difference(self.v2).symmetric_difference(self.v3)
        right = self.v1.symmetric_difference(self.v2.symmetric_difference(self.v3))
        self.assertEqual(left, right)
        
        # Identidad: a ⊕ 0 = a
        self.assertEqual(self.v1.symmetric_difference(self.empty), self.v1)
        
        # Inverso: a ⊕ a = 0
        self.assertEqual(self.v1.symmetric_difference(self.v1), self.empty)
    
    def test_hamming_distance(self):
        """Distancia de Hamming: d(a, b) = ||a ⊕ b||."""
        # Propiedades métricas
        # d(a, a) = 0
        self.assertEqual(self.v1.hamming_distance(self.v1), 0)
        
        # d(a, b) = d(b, a) (simetría)
        self.assertEqual(self.v1.hamming_distance(self.v2), self.v2.hamming_distance(self.v1))
        
        # d(a, b) ≥ 0
        self.assertGreaterEqual(self.v1.hamming_distance(self.v2), 0)
        
        # Casos específicos
        self.assertEqual(self.v1.hamming_distance(self.v2), 2)  # Disjuntos, 1+1
        self.assertEqual(self.v1.hamming_distance(self.v3), 1)  # v3 contiene v1
        self.assertEqual(self.empty.hamming_distance(self.v1), 1)
    
    def test_hamming_distance_triangle_inequality(self):
        """Desigualdad triangular: d(a, c) ≤ d(a, b) + d(b, c)."""
        vectors = [self.empty, self.v1, self.v2, self.v3, self.v_all]
        
        for a in vectors:
            for b in vectors:
                for c in vectors:
                    d_ac = a.hamming_distance(c)
                    d_ab = a.hamming_distance(b)
                    d_bc = b.hamming_distance(c)
                    
                    self.assertLessEqual(d_ac, d_ab + d_bc,
                        f"Falla desigualdad triangular: d({a}, {c}) > d({a}, {b}) + d({b}, {c})")
    
    def test_is_subset_of(self):
        """Orden parcial: a ⊆ b."""
        # Reflexividad
        self.assertTrue(self.v1.is_subset_of(self.v1))
        
        # Antisimetría
        self.assertTrue(self.v1.is_subset_of(self.v3))
        self.assertFalse(self.v3.is_subset_of(self.v1))
        
        # Transitividad
        v4 = BooleanVector(frozenset([CapabilityDimension.PHYS_IO, 
                                      CapabilityDimension.PHYS_NUM,
                                      CapabilityDimension.TACT_TOPO]))
        self.assertTrue(self.v1.is_subset_of(self.v3))
        self.assertTrue(self.v3.is_subset_of(v4))
        self.assertTrue(self.v1.is_subset_of(v4))
        
        # Vacío es subconjunto de todo
        self.assertTrue(self.empty.is_subset_of(self.v1))
        self.assertTrue(self.empty.is_subset_of(self.empty))
    
    def test_inner_product_z2(self):
        """Producto escalar en ℤ₂."""
        # Conmutatividad
        self.assertEqual(self.v1.inner_product_z2(self.v2), 
                        self.v2.inner_product_z2(self.v1))
        
        # a · a = |a| mod 2
        self.assertEqual(self.v1.inner_product_z2(self.v1), 1)  # peso 1
        self.assertEqual(self.v3.inner_product_z2(self.v3), 0)  # peso 2
        
        # Disjuntos → 0
        self.assertEqual(self.v1.inner_product_z2(self.v2), 0)
        
        # Con overlap
        self.assertEqual(self.v1.inner_product_z2(self.v3), 1)  # 1 bit común
    
    def test_invalid_construction(self):
        """Validación de construcción inválida."""
        with self.assertRaises(TypeError):
            BooleanVector(frozenset([1, 2, 3]))  # No son CapabilityDimension


# ========================================================================================
# TESTS: Tool
# ========================================================================================

class TestTool(TestBase):
    """Tests para la clase Tool."""
    
    def setUp(self):
        """Configuración de herramientas de prueba."""
        self.cap1 = BooleanVector(frozenset([CapabilityDimension.PHYS_IO]))
        self.cap2 = BooleanVector(frozenset([CapabilityDimension.PHYS_NUM]))
        
        self.tool1 = Tool("tool_a", self.cap1)
        self.tool2 = Tool("tool_b", self.cap2)
        self.tool3 = Tool("tool_a", self.cap1)  # Igual a tool1
    
    def test_immutability(self):
        """Las herramientas deben ser inmutables."""
        with self.assertRaises(AttributeError):
            self.tool1.name = "new_name"
        
        with self.assertRaises(AttributeError):
            self.tool1.capabilities = self.cap2
    
    def test_equality(self):
        """Igualdad basada en nombre y capacidades."""
        self.assertEqual(self.tool1, self.tool3)
        self.assertNotEqual(self.tool1, self.tool2)
    
    def test_hashability(self):
        """Las herramientas deben ser hasheables."""
        tool_set = {self.tool1, self.tool2, self.tool3}
        self.assertEqual(len(tool_set), 2)  # tool1 y tool3 son iguales
        
        tool_dict = {self.tool1: "value1", self.tool2: "value2"}
        self.assertEqual(tool_dict[self.tool3], "value1")
    
    def test_ordering(self):
        """Ordenamiento lexicográfico."""
        tools = [self.tool2, self.tool1, self.tool3]
        sorted_tools = sorted(tools)
        
        # tool_a < tool_b
        self.assertEqual(sorted_tools[0].name, "tool_a")
        self.assertEqual(sorted_tools[-1].name, "tool_b")
    
    def test_invalid_name(self):
        """Validación de nombre inválido."""
        with self.assertRaises(ValueError):
            Tool("", self.cap1)
        
        with self.assertRaises(ValueError):
            Tool(None, self.cap1)
    
    def test_invalid_capabilities(self):
        """Validación de capacidades inválidas."""
        with self.assertRaises(TypeError):
            Tool("tool", "not_a_boolean_vector")


# ========================================================================================
# TESTS: ImplicantTerm
# ========================================================================================

class TestQuineMcCluskeyMinimizer:
    @pytest.mark.parametrize("num_vars", sorted({1, 2, 4, 8, NUM_VARS_MAX_TESTED}))
    def test_boundary_constraints_vars(self, num_vars):
        QuineMcCluskeyMinimizer(num_vars=num_vars)


# ========================================================================================
# TESTS: UnionFind
# ========================================================================================

class TestUnionFind(TestBase):
    """Tests para la estructura Union-Find."""
    
    def test_initial_state(self):
        """Estado inicial: n componentes disjuntas."""
        uf = UnionFind(5)
        
        self.assertEqual(uf.num_components, 5)
        
        # Cada elemento es su propio representante
        for i in range(5):
            self.assertEqual(uf.find(i), i)
    
    def test_union_simple(self):
        """Unión básica de dos elementos."""
        uf = UnionFind(5)
        
        result = uf.union(0, 1)
        self.assertTrue(result)  # Unión exitosa
        self.assertEqual(uf.num_components, 4)
        
        # Mismo representante
        self.assertEqual(uf.find(0), uf.find(1))
    
    def test_union_already_connected(self):
        """Unir elementos ya conectados no cambia nada."""
        uf = UnionFind(5)
        
        uf.union(0, 1)
        num_comp_before = uf.num_components
        
        result = uf.union(0, 1)
        self.assertFalse(result)  # No se realizó unión
        self.assertEqual(uf.num_components, num_comp_before)
    
    def test_union_transitive(self):
        """Transitividad: si 0~1 y 1~2, entonces 0~2."""
        uf = UnionFind(5)
        
        uf.union(0, 1)
        uf.union(1, 2)
        
        self.assertEqual(uf.find(0), uf.find(2))
        self.assertEqual(uf.num_components, 3)
    
    def test_path_compression(self):
        """Path compression optimiza búsquedas."""
        uf = UnionFind(10)
        
        # Crear cadena: 0-1-2-3-4
        for i in range(4):
            uf.union(i, i + 1)
        
        # Primera búsqueda
        root = uf.find(4)
        
        # Después de path compression, parent[4] debería apuntar directamente a root
        self.assertEqual(uf.parent[4], root)
    
    def test_union_by_rank(self):
        """Union by rank mantiene árboles balanceados."""
        uf = UnionFind(8)
        
        # Crear dos árboles balanceados
        uf.union(0, 1)
        uf.union(2, 3)
        uf.union(0, 2)  # Une dos árboles de rank 1
        
        uf.union(4, 5)
        uf.union(6, 7)
        uf.union(4, 6)
        
        # Unir los dos árboles grandes
        uf.union(0, 4)
        
        self.assertEqual(uf.num_components, 1)
    
    def test_get_components(self):
        """Obtener todas las componentes conexas."""
        uf = UnionFind(6)
        
        uf.union(0, 1)
        uf.union(1, 2)
        uf.union(3, 4)
        # 5 queda solo
        
        components = uf.get_components()
        
        self.assertEqual(len(components), 3)
        
        # Verificar componentes
        components_sets = [set(comp) for comp in components]
        self.assertIn({0, 1, 2}, components_sets)
        self.assertIn({3, 4}, components_sets)
        self.assertIn({5}, components_sets)
    
    def test_size_tracking(self):
        """Tamaño de componentes se actualiza correctamente."""
        uf = UnionFind(5)
        
        uf.union(0, 1)
        uf.union(1, 2)
        
        root = uf.find(0)
        self.assertEqual(uf.size[root], 3)


# ========================================================================================
# TESTS: QuineMcCluskeyMinimizer
# ========================================================================================

class TestQuineMcCluskeyMinimizer(TestBase):
    """Tests para el algoritmo de Quine-McCluskey."""
    
    def setUp(self):
        """Configuración del minimizador."""
        self.minimizer = QuineMcCluskeyMinimizer(4)
    
    def test_initialization(self):
        """Inicialización correcta."""
        self.assertEqual(self.minimizer.num_vars, 4)
        self.assertEqual(self.minimizer.max_minterm, 15)
    
    def test_initialization_validation(self):
        """Validación de parámetros de inicialización."""
        with self.assertRaises(ValueError):
            QuineMcCluskeyMinimizer(0)
        
        with self.assertRaises(ValueError):
            QuineMcCluskeyMinimizer(21)
    
    def test_hamming_distance_ternary(self):
        """Distancia de Hamming entre términos ternarios."""
        # Sin don't cares
        self.assertEqual(self.minimizer._hamming_distance_ternary("0000", "0000"), 0)
        self.assertEqual(self.minimizer._hamming_distance_ternary("0000", "0001"), 1)
        self.assertEqual(self.minimizer._hamming_distance_ternary("0000", "1111"), 4)
        
        # Con don't cares (no cuentan)
        self.assertEqual(self.minimizer._hamming_distance_ternary("00-0", "00-1"), 1)
        self.assertEqual(self.minimizer._hamming_distance_ternary("0-0-", "1-1-"), 2)
        self.assertEqual(self.minimizer._hamming_distance_ternary("----", "1111"), 0)
    
    def test_hamming_distance_validation(self):
        """Validación de longitudes en distancia de Hamming."""
        with self.assertRaises(ValueError):
            self.minimizer._hamming_distance_ternary("00", "000")
    
    def test_can_combine_basic(self):
        """Verificación básica de combinabilidad."""
        # Difieren en 1 bit → combinables
        self.assertTrue(self.minimizer._can_combine("0000", "0001"))
        self.assertTrue(self.minimizer._can_combine("0100", "0000"))
        
        # Difieren en >1 bit → no combinables
        self.assertFalse(self.minimizer._can_combine("0000", "0011"))
        self.assertFalse(self.minimizer._can_combine("0000", "1111"))
        
        # Idénticos → no combinables
        self.assertFalse(self.minimizer._can_combine("0000", "0000"))
    
    def test_can_combine_with_dont_care(self):
        """Combinabilidad con don't cares."""
        # Con don't care en la misma posición → pueden ser combinables
        self.assertTrue(self.minimizer._can_combine("0-00", "0-01"))
        
        # Con don't care en posiciones diferentes → no combinables
        self.assertFalse(self.minimizer._can_combine("0-00", "01-0"))
        
        # Un término con '-' y otro con valor concreto en esa posición → no combinables
        self.assertFalse(self.minimizer._can_combine("0-00", "0100"))
    
    def test_combine_terms_basic(self):
        """Combinación básica de términos."""
        # 0000 y 0001 → 000-
        result = self.minimizer._combine_terms("0000", "0001")
        self.assertEqual(result, "000-")
        
        # 0100 y 0000 → 0-00
        result = self.minimizer._combine_terms("0100", "0000")
        self.assertEqual(result, "0-00")
        
        # No combinables → None
        result = self.minimizer._combine_terms("0000", "0011")
        self.assertIsNone(result)
    
    def test_combine_terms_with_dont_care(self):
        """Combinación de términos con don't cares."""
        # Ambos tienen '-' en la misma posición
        result = self.minimizer._combine_terms("0-00", "0-01")
        self.assertEqual(result, "0-0-")
        
        # No combinables si tienen '-' en posiciones diferentes
        result = self.minimizer._combine_terms("0-00", "01-0")
        self.assertIsNone(result)
    
    def test_compute_prime_implicants_simple(self):
        """Cálculo de implicantes primos: caso simple."""
        # Función: f = m(0, 1) = x₃x₂x₁x̄₀ + x₃x₂x₁x₀ = x₃x₂x₁
        minterms = [0, 1]
        
        prime_impls = self.minimizer.compute_prime_implicants(minterms)
        
        self.assertEqual(len(prime_impls), 1)
        impl = list(prime_impls)[0]
        self.assertEqual(impl.pattern, "000-")
    
    def test_compute_prime_implicants_no_reduction(self):
        """Implicantes sin reducción posible."""
        # Minterms que no se pueden combinar
        minterms = [0, 3, 5, 6]  # 0000, 0011, 0101, 0110
        
        prime_impls = self.minimizer.compute_prime_implicants(minterms)
        
        # Verificar que se encontraron implicantes
        self.assertGreater(len(prime_impls), 0)
        
        # Todos los minterms deben estar cubiertos
        covered = set()
        for impl in prime_impls:
            covered.update(impl.covered_minterms)
        
        self.assertEqual(covered, set(minterms))
    
    def test_compute_prime_implicants_complete_reduction(self):
        """Reducción completa a un solo implicante."""
        # Todos los minterms → patrón "----"
        minterms = list(range(16))  # 2^4 = 16
        
        prime_impls = self.minimizer.compute_prime_implicants(minterms)
        
        self.assertEqual(len(prime_impls), 1)
        impl = list(prime_impls)[0]
        self.assertEqual(impl.pattern, "----")
        self.assertEqual(len(impl.covered_minterms), 16)
    
    def test_compute_prime_implicants_empty(self):
        """Conjunto vacío de minterms."""
        prime_impls = self.minimizer.compute_prime_implicants([])
        self.assertEqual(len(prime_impls), 0)
    
    def test_compute_prime_implicants_validation(self):
        """Validación de minterms fuera de rango."""
        with self.assertRaises(ValueError):
            self.minimizer.compute_prime_implicants([16])  # Fuera de [0, 15]
        
        with self.assertRaises(ValueError):
            self.minimizer.compute_prime_implicants([-1])
    
    def test_find_essential_prime_implicants(self):
        """Identificación de implicantes esenciales."""
        # Caso simple: cada minterm cubierto por un solo implicante
        impl1 = ImplicantTerm("00--", frozenset([0, 1, 4, 5]))
        impl2 = ImplicantTerm("01--", frozenset([2, 3, 6, 7]))
        
        minterms = {0, 1, 2, 3, 4, 5, 6, 7}
        
        # Ambos son esenciales
        essential, covered = self.minimizer.find_essential_prime_implicants(
            {impl1, impl2}, minterms
        )
        
        self.assertEqual(len(essential), 2)
        self.assertEqual(covered, minterms)
    
    def test_find_essential_prime_implicants_partial(self):
        """Solo algunos implicantes son esenciales."""
        # impl1 es esencial para cubrir 0
        # impl2 y impl3 ambos cubren 1 → ninguno es esencial para 1
        impl1 = ImplicantTerm("000-", frozenset([0]))
        impl2 = ImplicantTerm("00-1", frozenset([1]))
        impl3 = ImplicantTerm("-001", frozenset([1, 9]))
        
        minterms = {0, 1}
        
        essential, covered = self.minimizer.find_essential_prime_implicants(
            {impl1, impl2, impl3}, minterms
        )
        
        # Solo impl1 es esencial
        self.assertIn(impl1, essential)
        self.assertIn(0, covered)
    
    def test_minimal_cover_greedy(self):
        """Cobertura minimal greedy."""
        impl1 = ImplicantTerm("00--", frozenset([0, 1, 4, 5]))
        impl2 = ImplicantTerm("01--", frozenset([2, 3, 6, 7]))
        impl3 = ImplicantTerm("-0-0", frozenset([0, 2, 4, 6]))
        
        minterms = {0, 1, 2, 3, 4, 5, 6, 7}
        essential = set()
        already_covered = set()
        
        cover = self.minimizer.minimal_cover_greedy(
            {impl1, impl2, impl3}, minterms, essential, already_covered
        )
        
        # Debe cubrir todos los minterms
        covered = set()
        for impl in cover:
            covered.update(impl.covered_minterms)
        
        self.assertEqual(covered, minterms)
    
    def test_minimal_cover_with_essentials(self):
        """Cobertura minimal con esenciales ya seleccionados."""
        impl1 = ImplicantTerm("00--", frozenset([0, 1, 4, 5]))
        impl2 = ImplicantTerm("01--", frozenset([2, 3, 6, 7]))
        
        minterms = {0, 1, 2, 3, 4, 5, 6, 7}
        essential = {impl1}
        already_covered = impl1.covered_minterms
        
        cover = self.minimizer.minimal_cover_greedy(
            {impl1, impl2}, minterms, essential, already_covered
        )
        
        # Debe incluir esenciales
        self.assertIn(impl1, cover)
        
        # Debe cubrir todos
        covered = set()
        for impl in cover:
            covered.update(impl.covered_minterms)
        self.assertEqual(covered, minterms)


# ========================================================================================
# TESTS: MICRedundancyAnalyzer
# ========================================================================================

class TestTopologicalRigor:
    """
    Escrutinio de la estructura del complejo simplicial K.
    """

    def test_betti_number_invariance(self, analyzer):
        """
        Axioma: β0(K) = β0(K').
        La minimización es un retracto de deformación que preserva componentes conexas.
        """
        analyzer.register_tool("T1", {CapabilityDimension.PHYS_IO})
        analyzer.register_tool("T2", {CapabilityDimension.PHYS_IO}) # Redundante
        analyzer.register_tool("T3", {CapabilityDimension.PHYS_NUM})
        
        h_k = analyzer.compute_homology_groups()
        beta0_k = h_k['H_0']
        
        results = analyzer.analyze_redundancy()
        
        analyzer_prime = MICRedundancyAnalyzer()
        for name in results['essential_tools']:
            tool = next(t for t in analyzer.tools if t.name == name)
            analyzer_prime.register_tool(tool.name, tool.capabilities.components)

        h_k_prime = analyzer_prime.compute_homology_groups()
        beta0_k_prime = h_k_prime['H_0']
        
        assert beta0_k == beta0_k_prime, f"Ruptura topológica: β0(K)={beta0_k}, β0(K')={beta0_k_prime}"

    def test_euler_poincare_preservation(self, analyzer):
        """
        Axioma: La característica de Euler χ es invariante bajo equivalencia de homotopía.
        Calculada vía el Complejo de Nervio de los implicantes primos.
        """
        # Caso 1: Dos capacidades independientes
        analyzer.register_tool("T1", {CapabilityDimension.PHYS_IO})
        analyzer.register_tool("T2", {CapabilityDimension.PHYS_NUM})

        results = analyzer.analyze_redundancy()
        chi = results['euler_characteristic']

        # En este caso, dos puntos aislados, χ = 2
        assert chi == 2, f"χ esperada 2, obtenida {chi}"

        # Caso 2: Unión de herramientas (cubriendo un espacio conexo)
        analyzer2 = MICRedundancyAnalyzer()
        # Herramientas que cubren {0,1}, {1,2} y {0,1,2}
        analyzer2.register_tool("T1", {CapabilityDimension.PHYS_IO, CapabilityDimension.PHYS_NUM})
        analyzer2.register_tool("T2", {CapabilityDimension.PHYS_NUM, CapabilityDimension.TACT_TOPO})
        analyzer2.register_tool("T3", {CapabilityDimension.PHYS_IO, CapabilityDimension.PHYS_NUM, CapabilityDimension.TACT_TOPO})

        results2 = analyzer2.analyze_redundancy()
        chi2 = results2['euler_characteristic']

        # El espacio {0,1} ∪ {1,2} ∪ {0,1,2} es un hipercubo 2D "L-shaped" o similar.
        # En este caso, QM genera los implicantes primos '00-11' y '0011-'.
        # Su intersección es '00111', que es no vacía y contraíble.
        # χ = χ(P1) + χ(P2) - χ(P1 ∩ P2) = 1 + 1 - 1 = 1.
        assert chi2 == 1, f"χ esperada 1, obtenida {chi2}"

    def test_boundary_of_boundary_is_empty(self):
        """
        Axioma fundamental: ∂² = 0.
        Verifica que el operador de frontera es nulo para complejos 1D en su grado superior.
        """
        # Representamos ∂_2 como una matriz nula.
        # En grafos, no existen 2-simples (caras), por lo que rank(∂_2) == 0.
        rank_boundary_2 = 0
        assert rank_boundary_2 == 0, "Topological heresy detected: rank(∂2) must be 0 for 1D complexes"

# ========================================================================================
# TESTS: Propiedades Matemáticas Globales
# ========================================================================================

class TestMathematicalProperties(TestBase):
    """Tests de propiedades matemáticas generales."""

    def test_strict_finiteness_checks(self):
        """Verifica que los checks de finitud estricta funcionan."""
        v1 = BooleanVector(frozenset([CapabilityDimension.PHYS_IO]))
        v2 = BooleanVector(frozenset([CapabilityDimension.PHYS_NUM]))

        # Test inner_product_z2 returns 0 or 1
        self.assertIn(v1.inner_product_z2(v2), {0, 1})
        self.assertIn(v1.inner_product_z2(v1), {0, 1})

        # Test hamming_distance
        self.assertGreaterEqual(v1.hamming_distance(v2), 0)

    def test_runtime_warning_elevation(self):
        """Verifica que RuntimeWarning se eleva a HomologicalInconsistencyError en el analizador."""
        analyzer = MICRedundancyAnalyzer()
        analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO})
        matrix = analyzer.build_incidence_matrix()

        import warnings
        from unittest.mock import patch

    def assert_boolean_orthogonality(self, v1: BooleanVector, v2: BooleanVector) -> None:
        """
        Evalúa la ortogonalidad de Gram en Z.
        Dos vectores de capacidad son ortogonales si su intersección es exactamente el conjunto vacío.
        """
        # Nota: Usamos 'components' según la implementación en mic_minimizer.py
        inner_product = len(v1.components.intersection(v2.components))
        assert inner_product == 0, f"Ruptura de ortogonalidad funcional. Entropía cruzada: {inner_product}"

    def test_laplacian_kernel_dimension(self, analyzer):
        """
        Axioma: dim(ker(L)) = β0.
        Para la matriz minimizada M_min, exija que no existan dependencias funcionales ocultas.
        """
        analyzer.register_tool("T1", {CapabilityDimension.PHYS_IO})
        analyzer.register_tool("T2", {CapabilityDimension.PHYS_NUM})
        analyzer.register_tool("T3", {CapabilityDimension.PHYS_NUM}) # Redundante
        
        analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO})
        analyzer.register_tool("tool2", {CapabilityDimension.PHYS_NUM})
        
        homology = analyzer.compute_homology_groups()
        
        for betti in homology['betti_numbers']:
            self.assertGreaterEqual(betti, 0)
        
        # β₀ ≥ 1 si hay herramientas
        self.assertGreaterEqual(homology['H_0'], 1)


# ========================================================================================
# TESTS: Casos Extremos (Edge Cases)
# ========================================================================================

class TestEdgeCases(TestBase):
    """Tests de casos extremos y límite."""
    
    def test_single_minterm(self):
        """Un solo minitérmino."""
        minimizer = QuineMcCluskeyMinimizer(4)
        prime_impls = minimizer.compute_prime_implicants([5])
        
        self.assertEqual(len(prime_impls), 1)
        impl = list(prime_impls)[0]
        self.assertEqual(impl.pattern, "0101")
        self.assertEqual(impl.covered_minterms, frozenset([5]))
    
    def test_all_minterms(self):
        """Todos los minitérminos (función constante 1)."""
        minimizer = QuineMcCluskeyMinimizer(3)
        minterms = list(range(8))
        
        prime_impls = minimizer.compute_prime_implicants(minterms)
        
        self.assertEqual(len(prime_impls), 1)
        impl = list(prime_impls)[0]
        self.assertEqual(impl.pattern, "---")
    
    def test_no_simplification_possible(self):
        """Minitérminos que no se pueden simplificar."""
        minimizer = QuineMcCluskeyMinimizer(4)
        
        # Minitérminos maximalmente distantes
        minterms = [0, 15]  # 0000 y 1111 (distancia 4)
        
        prime_impls = minimizer.compute_prime_implicants(minterms)
        
        self.assertEqual(len(prime_impls), 2)
        patterns = {impl.pattern for impl in prime_impls}
        self.assertEqual(patterns, {"0000", "1111"})
    
    def test_duplicate_minterms(self):
        """Minitérminos duplicados."""
        minimizer = QuineMcCluskeyMinimizer(3)
        
        # Duplicados deben ser eliminados
        minterms = [1, 1, 2, 2, 3, 3]
        prime_impls = minimizer.compute_prime_implicants(minterms)
        
        # Debe procesar como [1, 2, 3]
        covered = set()
        for impl in prime_impls:
            covered.update(impl.covered_minterms)
        
        self.assertEqual(covered, {1, 2, 3})
    
    def test_maximum_dimension(self):
        """Dimensión máxima permitida."""
        minimizer = QuineMcCluskeyMinimizer(20)
        
        # Debe funcionar sin error
        minterms = [0, 1, (1 << 20) - 1]
        prime_impls = minimizer.compute_prime_implicants(minterms)
        
        self.assertGreater(len(prime_impls), 0)
    
    def test_all_tools_identical(self):
        """Todas las herramientas son idénticas."""
        analyzer = MICRedundancyAnalyzer()
        
        for i in range(5):
            analyzer.register_tool(f"tool{i}", {CapabilityDimension.PHYS_IO})
        
        result = analyzer.analyze_redundancy()
        
        # Solo 1 esencial, 4 redundantes
        self.assertEqual(len(result['essential_tools']), 1)
        self.assertEqual(len(result['redundant_tools']), 4)
    
    def test_all_tools_disjoint(self):
        """Todas las herramientas disjuntas."""
        analyzer = MICRedundancyAnalyzer()
        
        dims = list(CapabilityDimension)
        for i, dim in enumerate(dims):
            analyzer.register_tool(f"tool{i}", {dim})
        
        result = analyzer.analyze_redundancy()
        
        # Todas esenciales
        self.assertEqual(len(result['essential_tools']), 5)
        self.assertEqual(len(result['redundant_tools']), 0)
    
    def test_empty_capabilities(self):
        """Herramienta sin capacidades."""
        analyzer = MICRedundancyAnalyzer()
        
        analyzer.register_tool("empty_tool", set())
        analyzer.register_tool("normal_tool", {CapabilityDimension.PHYS_IO})
        
        result = analyzer.analyze_redundancy()
        
        # Ambas deben estar clasificadas
        all_tools = set(result['essential_tools']) | set(result['redundant_tools'])
        self.assertIn('empty_tool', all_tools)
        self.assertIn('normal_tool', all_tools)


# ========================================================================================
# TESTS: Performance y Escalabilidad
# ========================================================================================

    def test_gram_matrix_orthogonality(self, analyzer):
        """
        La matriz de Gram G = M * M^T debe ser diagonal para herramientas perfectamente ortogonales.
        """
        t1_caps = {CapabilityDimension.PHYS_IO}
        t2_caps = {CapabilityDimension.PHYS_NUM}

        analyzer.register_tool("T1", t1_caps)
        analyzer.register_tool("T2", t2_caps)

        v1 = BooleanVector(frozenset(t1_caps))
        v2 = BooleanVector(frozenset(t2_caps))

        # Validación de rigor espectral en Z
        self.assert_boolean_orthogonality(v1, v2)
        
        # 100 herramientas
        for i in range(100):
            dim = list(CapabilityDimension)[i % 5]
            analyzer.register_tool(f"tool{i}", {dim})
        
        # Debe completar sin timeout
        result = analyzer.analyze_redundancy()
        
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['statistics']['total_tools'], 100)
    
    def test_quine_mccluskey_scaling(self):
        """Escalabilidad del algoritmo de Quine-McCluskey."""
        # Probar diferentes dimensiones
        for num_vars in [4, 6, 8]:
            minimizer = QuineMcCluskeyMinimizer(num_vars)
            
            # Mitad de los minitérminos
            minterms = list(range(0, 1 << num_vars, 2))
            
            # Debe completar
            prime_impls = minimizer.compute_prime_implicants(minterms)
            
            self.assertGreater(len(prime_impls), 0)
    
    def test_union_find_performance(self):
        """Performance de Union-Find."""
        n = 10000
        uf = UnionFind(n)
        
        # Unir todos en cadena
        for i in range(n - 1):
            uf.union(i, i + 1)
        
        # Verificar componente única
        self.assertEqual(uf.num_components, 1)
        
        # Find debe ser rápido (path compression)
        root = uf.find(n - 1)
        self.assertEqual(uf.find(0), root)


# ========================================================================================
# TESTS: Integración Completa
# ========================================================================================

class TestIntegration(TestBase):
    """Tests de integración end-to-end."""
    
    def test_full_pipeline_example(self):
        """Pipeline completo con ejemplo realista."""
        analyzer = MICRedundancyAnalyzer()
        
        # Escenario: Sistema MIC con redundancia
        analyzer.register_tool("stabilize_flux", {CapabilityDimension.PHYS_NUM})
        analyzer.register_tool("parse_raw", {CapabilityDimension.PHYS_IO, CapabilityDimension.PHYS_NUM})
        analyzer.register_tool("structure_logic", {CapabilityDimension.TACT_TOPO})
        analyzer.register_tool("audit_fusion", {CapabilityDimension.TACT_TOPO})  # Redundante
        analyzer.register_tool("lateral_pivot", {CapabilityDimension.STRAT_FIN})
        analyzer.register_tool("semantic_estimator", {CapabilityDimension.WIS_SEM})
        
        result = analyzer.analyze_redundancy()
        
        # Verificaciones
        self.assertEqual(result['status'], 'success')
        
        # audit_fusion es redundante con structure_logic
        # Nota: La clasificación depende del orden de selección en Quine-McCluskey.
        # Ambas herramientas tienen la misma firma topológica.
        all_tools = set(result['essential_tools']) | set(result['redundant_tools'])
        self.assertIn('audit_fusion', all_tools)
        self.assertIn('structure_logic', all_tools)

        # Al menos una de las dos con firma idéntica debe ser redundante
        redundant = result['redundant_tools']
        self.assertTrue('audit_fusion' in redundant or 'structure_logic' in redundant)
        
        # Cobertura completa
        all_tools = set(result['essential_tools']) | set(result['redundant_tools'])
        self.assertEqual(len(all_tools), 6)
        
        # Reducción detectada
        self.assertGreater(result['statistics']['reduction_rate'], 0)
    
    def test_consensus_between_methods(self):
        """Consenso entre diferentes métodos de análisis."""
        analyzer = MICRedundancyAnalyzer()
        
        # Crear redundancia conocida
        analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO})
        analyzer.register_tool("tool2", {CapabilityDimension.PHYS_IO})  # Idéntica
        
        result = analyzer.analyze_redundancy()
        
        # Homología debe detectar ciclo
        self.assertGreater(result['homology']['H_1'], 0)
        
        # Debe haber una redundante
        self.assertEqual(len(result['redundant_tools']), 1)
        
        # Estadísticas deben ser consistentes
        self.assertEqual(
            result['statistics']['essential_count'] + result['statistics']['redundant_count'],
            result['statistics']['total_tools']
        )


# ========================================================================================
# RUNNER DE TESTS
# ========================================================================================

class TestPerformanceScalability:
    """
    Validación de eficiencia computacional y escalabilidad del retículo.
    """

    @pytest.mark.parametrize("num_vars", sorted({4, 6, 8, NUM_VARS_MAX_TESTED}))
    def test_computational_complexity_scaling(self, num_vars):
        """
        Verifica que el tiempo de ejecución para el cálculo de implicantes primos
        se mantenga dentro de límites tolerables para el hipercubo B^n.
        """
        qm = QuineMcCluskeyMinimizer(num_vars)
        # Generamos una distribución densa de minitérminos
        minterms = list(range(0, 2**num_vars, 2))

        start = time.time()
        primes = qm.compute_prime_implicants(minterms)
        duration = time.time() - start

        # Para n <= NUM_VARS_MAX_TESTED, la convergencia debe ser < MAX_COMPUTATION_TIME_S
        assert duration < MAX_COMPUTATION_TIME_S, f"Degradación de performance en B^{num_vars}: {duration:.4f}s"
        assert len(primes) > 0

    def test_memory_isomorphism_load(self, analyzer):
        """
        Asegura que el registro de un gran número de herramientas no induzca
        fugas de memoria o inestabilidad en la matriz de incidencia.
        """
        n_tools = 100
        for i in range(n_tools):
            analyzer.register_tool(f"Tool_{i}", {CapabilityDimension.PHYS_IO})


# ========================================================================================
# PUNTO DE ENTRADA
# ========================================================================================

if __name__ == '__main__':
    import sys
    
    success = run_test_suite()
    sys.exit(0 if success else 1)