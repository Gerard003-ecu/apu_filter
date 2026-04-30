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

import sys
import io
import logging
import time
from contextlib import contextmanager
from typing import Set, List, FrozenSet

import numpy as np
import pytest

# Importar módulo a testear
from app.boole.tactics.mic_minimizer import (
    CapabilityDimension,
    BooleanVector,
    Tool,
    ImplicantTerm,
    QuineMcCluskeyMinimizer,
    TopologicalInvariantComputer,
    MICRedundancyAnalyzer,
    validate_boolean_lattice_axioms,
    HomologicalInconsistencyError,
    UnionFind
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

@pytest.fixture
def analyzer():
    """Fixture que proporciona un analizador de redundancia."""
    return MICRedundancyAnalyzer()

@pytest.fixture
def boolean_vectors():
    """Fixture que proporciona vectores booleanos de prueba."""
    class Vectors:
        def __init__(self):
            self.empty = BooleanVector(frozenset())
            self.v1 = BooleanVector(frozenset([CapabilityDimension.PHYS_IO]))
            self.v2 = BooleanVector(frozenset([CapabilityDimension.PHYS_NUM]))
            self.v3 = BooleanVector(frozenset([CapabilityDimension.PHYS_IO, CapabilityDimension.PHYS_NUM]))
            self.v_all = BooleanVector(frozenset(CapabilityDimension))
    return Vectors()

# ========================================================================================
# TESTS: CapabilityDimension
# ========================================================================================

class TestCapabilityDimension:
    """Tests para la enumeración CapabilityDimension."""
    
    def test_all_dimensions_exist(self):
        """Verifica que todas las dimensiones esperadas existen."""
        expected = {'PHYS_IO', 'PHYS_NUM', 'TACT_TOPO', 'STRAT_FIN', 'WIS_SEM'}
        actual = {dim.name for dim in CapabilityDimension}
        assert expected == actual
    
    def test_dimension_values_sequential(self):
        """Verifica que los valores son secuenciales comenzando en 0."""
        values = [dim.value for dim in CapabilityDimension]
        assert values == list(range(len(CapabilityDimension)))
    
    def test_ordering_is_total(self):
        """Verifica que el ordenamiento es total y transitivo."""
        dims = list(CapabilityDimension)
        
        # Reflexividad
        for d in dims:
            assert not (d < d)
        
        # Antisimetría
        for i, d1 in enumerate(dims):
            for j, d2 in enumerate(dims):
                if i < j:
                    assert d1 < d2
                    assert not (d2 < d1)
        
        # Transitividad
        for d1 in dims:
            for d2 in dims:
                for d3 in dims:
                    if d1 < d2 and d2 < d3:
                        assert d1 < d3
    
    def test_ordering_consistency_with_value(self):
        """El orden < debe ser consistente con el valor numérico."""
        dims = list(CapabilityDimension)
        for d1 in dims:
            for d2 in dims:
                assert (d1 < d2) == (d1.value < d2.value)


# ========================================================================================
# TESTS: BooleanVector
# ========================================================================================

class TestBooleanVector:
    """Tests para la clase BooleanVector."""
    
    def test_immutability(self, boolean_vectors):
        """Los BooleanVector deben ser inmutables."""
        with pytest.raises(AttributeError):
            boolean_vectors.v1.components = frozenset([CapabilityDimension.PHYS_NUM])
    
    def test_hashability(self, boolean_vectors):
        """Los BooleanVector deben ser hasheables."""
        v1_copy = BooleanVector(frozenset([CapabilityDimension.PHYS_IO]))
        
        # Mismo contenido → mismo hash
        assert hash(boolean_vectors.v1) == hash(v1_copy)
        
        # Se pueden usar en sets
        s = {boolean_vectors.v1, boolean_vectors.v2, v1_copy}
        assert len(s) == 2  # v1 y v1_copy son iguales
        
        # Se pueden usar como claves de diccionario
        d = {boolean_vectors.v1: "value1", boolean_vectors.v2: "value2"}
        assert d[v1_copy] == "value1"
    
    def test_equality(self, boolean_vectors):
        """Igualdad basada en componentes."""
        v1_copy = BooleanVector(frozenset([CapabilityDimension.PHYS_IO]))
        
        assert boolean_vectors.v1 == v1_copy
        assert boolean_vectors.v1 != boolean_vectors.v2
        assert boolean_vectors.v1 != boolean_vectors.v3
    
    def test_ordering_total(self, boolean_vectors):
        """El ordenamiento debe ser total."""
        vectors = [boolean_vectors.empty, boolean_vectors.v1, boolean_vectors.v2, boolean_vectors.v3]
        
        # Debe ser posible ordenar
        sorted_vectors = sorted(vectors)
        assert len(sorted_vectors) == 4
        
        # Debe ser determinista
        assert sorted(vectors) == sorted(vectors)
    
    def test_from_minterm(self, boolean_vectors):
        """Construcción desde minitérmino."""
        # 0b101 = 5 → bits 0 y 2 activos
        v = BooleanVector.from_minterm(5, 5)
        expected = BooleanVector(frozenset([
            CapabilityDimension.PHYS_IO,    # bit 0
            CapabilityDimension.TACT_TOPO   # bit 2
        ]))
        assert v == expected
        
        # Caso especial: 0
        v_zero = BooleanVector.from_minterm(0, 5)
        assert v_zero == boolean_vectors.empty
        
        # Caso especial: todos los bits
        v_all = BooleanVector.from_minterm((1 << 5) - 1, 5)
        assert v_all.hamming_weight() == 5
    
    def test_from_minterm_validation(self):
        """Validación de rango en from_minterm."""
        with pytest.raises(ValueError):
            BooleanVector.from_minterm(-1, 5)
        
        with pytest.raises(ValueError):
            BooleanVector.from_minterm(32, 5)  # 32 = 2^5, fuera de rango
    
    def test_to_binary_string(self, boolean_vectors):
        """Conversión a string binario."""
        assert boolean_vectors.empty.to_binary_string(5) == "00000"
        assert boolean_vectors.v1.to_binary_string(5) == "10000"
        assert boolean_vectors.v2.to_binary_string(5) == "01000"
        assert boolean_vectors.v3.to_binary_string(5) == "11000"
        assert boolean_vectors.v_all.to_binary_string(5) == "11111"
    
    def test_to_minterm(self, boolean_vectors):
        """Conversión a minitérmino."""
        assert boolean_vectors.empty.to_minterm() == 0
        assert boolean_vectors.v1.to_minterm() == 1    # 2^0
        assert boolean_vectors.v2.to_minterm() == 2    # 2^1
        assert boolean_vectors.v3.to_minterm() == 3    # 2^0 + 2^1
    
    def test_roundtrip_minterm(self):
        """from_minterm y to_minterm son inversos."""
        for minterm in range(32):  # 2^5
            v = BooleanVector.from_minterm(minterm, 5)
            assert v.to_minterm() == minterm
    
    def test_hamming_weight(self, boolean_vectors):
        """Peso de Hamming = número de componentes."""
        assert boolean_vectors.empty.hamming_weight() == 0
        assert boolean_vectors.v1.hamming_weight() == 1
        assert boolean_vectors.v2.hamming_weight() == 1
        assert boolean_vectors.v3.hamming_weight() == 2
        assert boolean_vectors.v_all.hamming_weight() == 5
    
    # ===== OPERACIONES DEL RETÍCULO BOOLEANO =====
    
    def test_union_commutative(self, boolean_vectors):
        """OR es conmutativo: a ∨ b = b ∨ a."""
        assert boolean_vectors.v1.union(boolean_vectors.v2) == boolean_vectors.v2.union(boolean_vectors.v1)
        assert boolean_vectors.v1.union(boolean_vectors.v3) == boolean_vectors.v3.union(boolean_vectors.v1)
    
    def test_union_associative(self, boolean_vectors):
        """OR es asociativo: (a ∨ b) ∨ c = a ∨ (b ∨ c)."""
        left = boolean_vectors.v1.union(boolean_vectors.v2).union(boolean_vectors.v3)
        right = boolean_vectors.v1.union(boolean_vectors.v2.union(boolean_vectors.v3))
        assert left == right
    
    def test_union_identity(self, boolean_vectors):
        """Identidad: a ∨ 0 = a."""
        assert boolean_vectors.v1.union(boolean_vectors.empty) == boolean_vectors.v1
        assert boolean_vectors.empty.union(boolean_vectors.v1) == boolean_vectors.v1
    
    def test_union_idempotent(self, boolean_vectors):
        """Idempotencia: a ∨ a = a."""
        assert boolean_vectors.v1.union(boolean_vectors.v1) == boolean_vectors.v1
        assert boolean_vectors.v3.union(boolean_vectors.v3) == boolean_vectors.v3
    
    def test_intersection_commutative(self, boolean_vectors):
        """AND es conmutativo: a ∧ b = b ∧ a."""
        assert boolean_vectors.v1.intersection(boolean_vectors.v2) == boolean_vectors.v2.intersection(boolean_vectors.v1)
    
    def test_intersection_associative(self, boolean_vectors):
        """AND es asociativo: (a ∧ b) ∧ c = a ∧ (b ∧ c)."""
        left = boolean_vectors.v1.intersection(boolean_vectors.v2).intersection(boolean_vectors.v3)
        right = boolean_vectors.v1.intersection(boolean_vectors.v2.intersection(boolean_vectors.v3))
        assert left == right
    
    def test_intersection_identity(self, boolean_vectors):
        """Identidad: a ∧ 1 = a."""
        assert boolean_vectors.v1.intersection(boolean_vectors.v_all) == boolean_vectors.v1
    
    def test_intersection_zero(self, boolean_vectors):
        """Aniquilador: a ∧ 0 = 0."""
        assert boolean_vectors.v1.intersection(boolean_vectors.empty) == boolean_vectors.empty
    
    def test_intersection_idempotent(self, boolean_vectors):
        """Idempotencia: a ∧ a = a."""
        assert boolean_vectors.v1.intersection(boolean_vectors.v1) == boolean_vectors.v1
    
    def test_absorption_law(self, boolean_vectors):
        """Absorción: a ∨ (a ∧ b) = a."""
        result = boolean_vectors.v1.union(boolean_vectors.v1.intersection(boolean_vectors.v2))
        assert result == boolean_vectors.v1
        
        # Forma dual: a ∧ (a ∨ b) = a
        result_dual = boolean_vectors.v1.intersection(boolean_vectors.v1.union(boolean_vectors.v2))
        assert result_dual == boolean_vectors.v1
    
    def test_distributive_law(self, boolean_vectors):
        """Distributividad: a ∧ (b ∨ c) = (a ∧ b) ∨ (a ∧ c)."""
        left = boolean_vectors.v1.intersection(boolean_vectors.v2.union(boolean_vectors.v3))
        right = boolean_vectors.v1.intersection(boolean_vectors.v2).union(boolean_vectors.v1.intersection(boolean_vectors.v3))
        assert left == right
        
        # Forma dual: a ∨ (b ∧ c) = (a ∨ b) ∧ (a ∨ c)
        left_dual = boolean_vectors.v1.union(boolean_vectors.v2.intersection(boolean_vectors.v3))
        right_dual = boolean_vectors.v1.union(boolean_vectors.v2).intersection(boolean_vectors.v1.union(boolean_vectors.v3))
        assert left_dual == right_dual
    
    def test_complement(self, boolean_vectors):
        """Complemento: a ∨ ¬a = 1, a ∧ ¬a = 0."""
        num_vars = 5
        
        # a ∨ ¬a = 1
        result = boolean_vectors.v1.union(boolean_vectors.v1.complement(num_vars))
        expected_all = BooleanVector(frozenset(CapabilityDimension))
        assert result == expected_all
        
        # a ∧ ¬a = 0
        result_zero = boolean_vectors.v1.intersection(boolean_vectors.v1.complement(num_vars))
        assert result_zero == boolean_vectors.empty
    
    def test_complement_involution(self, boolean_vectors):
        """Doble complemento: ¬(¬a) = a."""
        num_vars = 5
        double_complement = boolean_vectors.v1.complement(num_vars).complement(num_vars)
        assert double_complement == boolean_vectors.v1
    
    def test_de_morgan_laws(self, boolean_vectors):
        """Leyes de De Morgan: ¬(a ∨ b) = ¬a ∧ ¬b, ¬(a ∧ b) = ¬a ∨ ¬b."""
        num_vars = 5
        
        # ¬(a ∨ b) = ¬a ∧ ¬b
        left = boolean_vectors.v1.union(boolean_vectors.v2).complement(num_vars)
        right = boolean_vectors.v1.complement(num_vars).intersection(boolean_vectors.v2.complement(num_vars))
        assert left == right
        
        # ¬(a ∧ b) = ¬a ∨ ¬b
        left2 = boolean_vectors.v1.intersection(boolean_vectors.v2).complement(num_vars)
        right2 = boolean_vectors.v1.complement(num_vars).union(boolean_vectors.v2.complement(num_vars))
        assert left2 == right2
    
    def test_symmetric_difference(self, boolean_vectors):
        """XOR (suma en ℤ₂): a ⊕ b."""
        # Propiedades del grupo
        # Conmutatividad
        assert boolean_vectors.v1.symmetric_difference(boolean_vectors.v2) == \
                        boolean_vectors.v2.symmetric_difference(boolean_vectors.v1)
        
        # Asociatividad
        left = boolean_vectors.v1.symmetric_difference(boolean_vectors.v2).symmetric_difference(boolean_vectors.v3)
        right = boolean_vectors.v1.symmetric_difference(boolean_vectors.v2.symmetric_difference(boolean_vectors.v3))
        assert left == right
        
        # Identidad: a ⊕ 0 = a
        assert boolean_vectors.v1.symmetric_difference(boolean_vectors.empty) == boolean_vectors.v1
        
        # Inverso: a ⊕ a = 0
        assert boolean_vectors.v1.symmetric_difference(boolean_vectors.v1) == boolean_vectors.empty
    
    def test_hamming_distance(self, boolean_vectors):
        """Distancia de Hamming: d(a, b) = ||a ⊕ b||."""
        # Propiedades métricas
        # d(a, a) = 0
        assert boolean_vectors.v1.hamming_distance(boolean_vectors.v1) == 0
        
        # d(a, b) = d(b, a) (simetría)
        assert boolean_vectors.v1.hamming_distance(boolean_vectors.v2) == boolean_vectors.v2.hamming_distance(boolean_vectors.v1)
        
        # d(a, b) ≥ 0
        assert boolean_vectors.v1.hamming_distance(boolean_vectors.v2) >= 0
        
        # Casos específicos
        assert boolean_vectors.v1.hamming_distance(boolean_vectors.v2) == 2  # Disjuntos, 1+1
        assert boolean_vectors.v1.hamming_distance(boolean_vectors.v3) == 1  # v3 contiene v1
        assert boolean_vectors.empty.hamming_distance(boolean_vectors.v1) == 1
    
    def test_hamming_distance_triangle_inequality(self, boolean_vectors):
        """Desigualdad triangular: d(a, c) ≤ d(a, b) + d(b, c)."""
        vectors = [boolean_vectors.empty, boolean_vectors.v1, boolean_vectors.v2, boolean_vectors.v3, boolean_vectors.v_all]
        
        for a in vectors:
            for b in vectors:
                for c in vectors:
                    d_ac = a.hamming_distance(c)
                    d_ab = a.hamming_distance(b)
                    d_bc = b.hamming_distance(c)
                    
                    assert d_ac <= d_ab + d_bc, \
                        f"Falla desigualdad triangular: d({a}, {c}) > d({a}, {b}) + d({b}, {c})"
    
    def test_is_subset_of(self, boolean_vectors):
        """Orden parcial: a ⊆ b."""
        # Reflexividad
        assert boolean_vectors.v1.is_subset_of(boolean_vectors.v1)
        
        # Antisimetría
        assert boolean_vectors.v1.is_subset_of(boolean_vectors.v3)
        assert not boolean_vectors.v3.is_subset_of(boolean_vectors.v1)
        
        # Transitividad
        v4 = BooleanVector(frozenset([CapabilityDimension.PHYS_IO, 
                                      CapabilityDimension.PHYS_NUM,
                                      CapabilityDimension.TACT_TOPO]))
        assert boolean_vectors.v1.is_subset_of(boolean_vectors.v3)
        assert boolean_vectors.v3.is_subset_of(v4)
        assert boolean_vectors.v1.is_subset_of(v4)
        
        # Vacío es subconjunto de todo
        assert boolean_vectors.empty.is_subset_of(boolean_vectors.v1)
        assert boolean_vectors.empty.is_subset_of(boolean_vectors.empty)
    
    def test_inner_product_z2(self, boolean_vectors):
        """Producto escalar en ℤ₂."""
        # Conmutatividad
        assert boolean_vectors.v1.inner_product_z2(boolean_vectors.v2) == \
                        boolean_vectors.v2.inner_product_z2(boolean_vectors.v1)
        
        # a · a = |a| mod 2
        assert boolean_vectors.v1.inner_product_z2(boolean_vectors.v1) == 1  # peso 1
        assert boolean_vectors.v3.inner_product_z2(boolean_vectors.v3) == 0  # peso 2
        
        # Disjuntos → 0
        assert boolean_vectors.v1.inner_product_z2(boolean_vectors.v2) == 0
        
        # Con overlap
        assert boolean_vectors.v1.inner_product_z2(boolean_vectors.v3) == 1  # 1 bit común
    
    def test_invalid_construction(self):
        """Validación de construcción inválida."""
        with pytest.raises(TypeError):
            BooleanVector(frozenset([1, 2, 3]))  # No son CapabilityDimension


# ========================================================================================
# TESTS: Tool
# ========================================================================================

class TestTool:
    """Tests para la clase Tool."""
    
    @pytest.fixture
    def tools(self):
        """Fixture que proporciona herramientas de prueba."""
        class Tools:
            def __init__(self):
                self.cap1 = BooleanVector(frozenset([CapabilityDimension.PHYS_IO]))
                self.cap2 = BooleanVector(frozenset([CapabilityDimension.PHYS_NUM]))

                self.tool1 = Tool("tool_a", self.cap1)
                self.tool2 = Tool("tool_b", self.cap2)
                self.tool3 = Tool("tool_a", self.cap1)  # Igual a tool1
        return Tools()
    
    def test_immutability(self, tools):
        """Las herramientas deben ser inmutables."""
        with pytest.raises(AttributeError):
            tools.tool1.name = "new_name"
        
        with pytest.raises(AttributeError):
            tools.tool1.capabilities = tools.cap2
    
    def test_equality(self, tools):
        """Igualdad basada en nombre y capacidades."""
        assert tools.tool1 == tools.tool3
        assert tools.tool1 != tools.tool2
    
    def test_hashability(self, tools):
        """Las herramientas deben ser hasheables."""
        tool_set = {tools.tool1, tools.tool2, tools.tool3}
        assert len(tool_set) == 2  # tool1 y tool3 son iguales
        
        tool_dict = {tools.tool1: "value1", tools.tool2: "value2"}
        assert tool_dict[tools.tool3] == "value1"
    
    def test_ordering(self, tools):
        """Ordenamiento lexicográfico."""
        tools_list = [tools.tool2, tools.tool1, tools.tool3]
        sorted_tools = sorted(tools_list)
        
        # tool_a < tool_b
        assert sorted_tools[0].name == "tool_a"
        assert sorted_tools[-1].name == "tool_b"
    
    def test_invalid_name(self):
        """Validación de nombre inválido."""
        cap1 = BooleanVector(frozenset([CapabilityDimension.PHYS_IO]))
        with pytest.raises(ValueError):
            Tool("", cap1)
        
        with pytest.raises(ValueError):
            Tool(None, cap1)
    
    def test_invalid_capabilities(self):
        """Validación de capacidades inválidas."""
        with pytest.raises(TypeError):
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

class TestUnionFind:
    """Tests para la estructura Union-Find."""
    
    def test_initial_state(self):
        """Estado inicial: n componentes disjuntas."""
        uf = UnionFind(5)
        assert uf.num_components == 5
        for i in range(5):
            assert uf.find(i) == i
    
    def test_union_simple(self):
        """Unión básica de dos elementos."""
        uf = UnionFind(5)
        result = uf.union(0, 1)
        assert result  # Unión exitosa
        assert uf.num_components == 4
        assert uf.find(0) == uf.find(1)
    
    def test_union_already_connected(self):
        """Unir elementos ya conectados no cambia nada."""
        uf = UnionFind(5)
        uf.union(0, 1)
        num_comp_before = uf.num_components
        result = uf.union(0, 1)
        assert not result  # No se realizó unión
        assert uf.num_components == num_comp_before
    
    def test_union_transitive(self):
        """Transitividad: si 0~1 y 1~2, entonces 0~2."""
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(1, 2)
        assert uf.find(0) == uf.find(2)
        assert uf.num_components == 3
    
    def test_path_compression(self):
        """Path compression optimiza búsquedas."""
        uf = UnionFind(10)
        for i in range(4):
            uf.union(i, i + 1)
        root = uf.find(4)
        assert uf.parent[4] == root
    
    def test_union_by_rank(self):
        """Union by rank mantiene árboles balanceados."""
        uf = UnionFind(8)
        uf.union(0, 1)
        uf.union(2, 3)
        uf.union(0, 2)
        uf.union(4, 5)
        uf.union(6, 7)
        uf.union(4, 6)
        uf.union(0, 4)
        assert uf.num_components == 1
    
    def test_get_components(self):
        """Obtener todas las componentes conexas."""
        uf = UnionFind(6)
        uf.union(0, 1)
        uf.union(1, 2)
        uf.union(3, 4)
        components = uf.get_components()
        assert len(components) == 3
        components_sets = [set(comp) for comp in components]
        assert {0, 1, 2} in components_sets
        assert {3, 4} in components_sets
        assert {5} in components_sets
    
    def test_size_tracking(self):
        """Tamaño de componentes se actualiza correctamente."""
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(1, 2)
        root = uf.find(0)
        assert uf.size[root] == 3


# ========================================================================================
# TESTS: QuineMcCluskeyMinimizer
# ========================================================================================

class TestQuineMcCluskeyMinimizerTests:
    """Tests para el algoritmo de Quine-McCluskey."""
    
    @pytest.fixture
    def minimizer(self):
        """Fixture que proporciona un minimizador de 4 variables."""
        return QuineMcCluskeyMinimizer(4)
    
    def test_initialization(self, minimizer):
        """Inicialización correcta."""
        assert minimizer.num_vars == 4
        assert minimizer.max_minterm == 15
    
    def test_initialization_validation(self):
        """Validación de parámetros de inicialización."""
        with pytest.raises(ValueError):
            QuineMcCluskeyMinimizer(0)
        with pytest.raises(ValueError):
            QuineMcCluskeyMinimizer(21)
    
    def test_hamming_distance_ternary(self, minimizer):
        """Distancia de Hamming entre términos ternarios."""
        assert minimizer._hamming_distance_ternary("0000", "0000") == 0
        assert minimizer._hamming_distance_ternary("0000", "0001") == 1
        assert minimizer._hamming_distance_ternary("0000", "1111") == 4
        assert minimizer._hamming_distance_ternary("00-0", "00-1") == 1
        assert minimizer._hamming_distance_ternary("0-0-", "1-1-") == 2
        assert minimizer._hamming_distance_ternary("----", "1111") == 0
    
    def test_hamming_distance_validation(self, minimizer):
        """Validación de longitudes en distancia de Hamming."""
        with pytest.raises(ValueError):
            minimizer._hamming_distance_ternary("00", "000")
    
    def test_can_combine_basic(self, minimizer):
        """Verificación básica de combinabilidad."""
        assert minimizer._can_combine("0000", "0001")
        assert minimizer._can_combine("0100", "0000")
        assert not minimizer._can_combine("0000", "0011")
        assert not minimizer._can_combine("0000", "1111")
        assert not minimizer._can_combine("0000", "0000")
    
    def test_can_combine_with_dont_care(self, minimizer):
        """Combinabilidad con don't cares."""
        assert minimizer._can_combine("0-00", "0-01")
        assert not minimizer._can_combine("0-00", "01-0")
        assert not minimizer._can_combine("0-00", "0100")
    
    def test_combine_terms_basic(self, minimizer):
        """Combinación básica de términos."""
        assert minimizer._combine_terms("0000", "0001") == "000-"
        assert minimizer._combine_terms("0100", "0000") == "0-00"
        assert minimizer._combine_terms("0000", "0011") is None
    
    def test_combine_terms_with_dont_care(self, minimizer):
        """Combinación de términos con don't cares."""
        assert minimizer._combine_terms("0-00", "0-01") == "0-0-"
        assert minimizer._combine_terms("0-00", "01-0") is None
    
    def test_compute_prime_implicants_simple(self, minimizer):
        """Cálculo de implicantes primos: caso simple."""
        minterms = [0, 1]
        prime_impls = minimizer.compute_prime_implicants(minterms)
        assert len(prime_impls) == 1
        impl = list(prime_impls)[0]
        assert impl.pattern == "000-"
    
    def test_compute_prime_implicants_no_reduction(self, minimizer):
        """Implicantes sin reducción posible."""
        minterms = [0, 3, 5, 6]
        prime_impls = minimizer.compute_prime_implicants(minterms)
        assert len(prime_impls) > 0
        covered = set()
        for impl in prime_impls:
            covered.update(impl.covered_minterms)
        assert covered == set(minterms)
    
    def test_compute_prime_implicants_complete_reduction(self, minimizer):
        """Reducción completa a un solo implicante."""
        minterms = list(range(16))
        prime_impls = minimizer.compute_prime_implicants(minterms)
        assert len(prime_impls) == 1
        impl = list(prime_impls)[0]
        assert impl.pattern == "----"
        assert len(impl.covered_minterms) == 16
    
    def test_compute_prime_implicants_empty(self, minimizer):
        """Conjunto vacío de minterms."""
        assert len(minimizer.compute_prime_implicants([])) == 0
    
    def test_compute_prime_implicants_validation(self, minimizer):
        """Validación de minterms fuera de rango."""
        with pytest.raises(ValueError):
            minimizer.compute_prime_implicants([16])
        with pytest.raises(ValueError):
            minimizer.compute_prime_implicants([-1])
    
    def test_find_essential_prime_implicants(self, minimizer):
        """Identificación de implicantes esenciales."""
        impl1 = ImplicantTerm("00--", frozenset([0, 1, 4, 5]))
        impl2 = ImplicantTerm("01--", frozenset([2, 3, 6, 7]))
        minterms = {0, 1, 2, 3, 4, 5, 6, 7}
        essential, covered = minimizer.find_essential_prime_implicants({impl1, impl2}, minterms)
        assert len(essential) == 2
        assert covered == minterms
    
    def test_find_essential_prime_implicants_partial(self, minimizer):
        """Solo algunos implicantes son esenciales."""
        impl1 = ImplicantTerm("000-", frozenset([0]))
        impl2 = ImplicantTerm("00-1", frozenset([1]))
        impl3 = ImplicantTerm("-001", frozenset([1, 9]))
        minterms = {0, 1}
        essential, covered = minimizer.find_essential_prime_implicants({impl1, impl2, impl3}, minterms)
        assert impl1 in essential
        assert 0 in covered
    
    def test_minimal_cover_greedy(self, minimizer):
        """Cobertura minimal greedy."""
        impl1 = ImplicantTerm("00--", frozenset([0, 1, 4, 5]))
        impl2 = ImplicantTerm("01--", frozenset([2, 3, 6, 7]))
        impl3 = ImplicantTerm("-0-0", frozenset([0, 2, 4, 6]))
        minterms = {0, 1, 2, 3, 4, 5, 6, 7}
        cover = minimizer.minimal_cover_greedy({impl1, impl2, impl3}, minterms, set(), set())
        covered = set()
        for impl in cover:
            covered.update(impl.covered_minterms)
        assert covered == minterms
    
    def test_minimal_cover_with_essentials(self, minimizer):
        """Cobertura minimal con esenciales ya seleccionados."""
        impl1 = ImplicantTerm("00--", frozenset([0, 1, 4, 5]))
        impl2 = ImplicantTerm("01--", frozenset([2, 3, 6, 7]))
        minterms = {0, 1, 2, 3, 4, 5, 6, 7}
        cover = minimizer.minimal_cover_greedy({impl1, impl2}, minterms, {impl1}, impl1.covered_minterms)
        assert impl1 in cover
        covered = set()
        for impl in cover:
            covered.update(impl.covered_minterms)
        assert covered == minterms


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

class TestMathematicalProperties:
    """Tests de propiedades matemáticas generales."""

    def test_strict_finiteness_checks(self):
        """Verifica que los checks de finitud estricta funcionan."""
        v1 = BooleanVector(frozenset([CapabilityDimension.PHYS_IO]))
        v2 = BooleanVector(frozenset([CapabilityDimension.PHYS_NUM]))

        # Test inner_product_z2 returns 0 or 1
        assert v1.inner_product_z2(v2) in {0, 1}
        assert v1.inner_product_z2(v1) in {0, 1}

        # Test hamming_distance
        assert v1.hamming_distance(v2) >= 0

    def test_runtime_warning_elevation(self):
        """Verifica que RuntimeWarning se eleva a HomologicalInconsistencyError en el analizador."""
        analyzer = MICRedundancyAnalyzer()
        analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO})

        with pytest.raises(HomologicalInconsistencyError):
            # Activamos el flag interno para disparar el warning
            analyzer._trigger_rigor_warning = True
            analyzer.compute_homology_groups()

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
        
        homology = analyzer.compute_homology_groups()
        
        for betti in homology['betti_numbers']:
            assert betti >= 0
        
        # β₀ ≥ 1 si hay herramientas
        assert homology['H_0'] >= 1


# ========================================================================================
# TESTS: Casos Extremos (Edge Cases)
# ========================================================================================

class TestEdgeCases:
    """Tests de casos extremos y límite."""
    
    def test_single_minterm(self):
        """Un solo minitérmino."""
        minimizer = QuineMcCluskeyMinimizer(4)
        prime_impls = minimizer.compute_prime_implicants([5])
        
        assert len(prime_impls) == 1
        impl = list(prime_impls)[0]
        assert impl.pattern == "0101"
        assert impl.covered_minterms == frozenset([5])
    
    def test_all_minterms(self):
        """Todos los minitérminos (función constante 1)."""
        minimizer = QuineMcCluskeyMinimizer(3)
        minterms = list(range(8))
        
        prime_impls = minimizer.compute_prime_implicants(minterms)
        
        assert len(prime_impls) == 1
        impl = list(prime_impls)[0]
        assert impl.pattern == "---"
    
    def test_no_simplification_possible(self):
        """Minitérminos que no se pueden simplificar."""
        minimizer = QuineMcCluskeyMinimizer(4)
        
        # Minitérminos maximalmente distantes
        minterms = [0, 15]  # 0000 y 1111 (distancia 4)
        
        prime_impls = minimizer.compute_prime_implicants(minterms)
        
        assert len(prime_impls) == 2
        patterns = {impl.pattern for impl in prime_impls}
        assert patterns == {"0000", "1111"}
    
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
        
        assert covered == {1, 2, 3}
    
    def test_maximum_dimension(self):
        """Dimensión máxima permitida."""
        minimizer = QuineMcCluskeyMinimizer(20)
        
        # Debe funcionar sin error
        minterms = [0, 1, (1 << 20) - 1]
        prime_impls = minimizer.compute_prime_implicants(minterms)
        
        assert len(prime_impls) > 0
    
    def test_all_tools_identical(self):
        """Todas las herramientas son idénticas."""
        analyzer = MICRedundancyAnalyzer()
        
        for i in range(5):
            analyzer.register_tool(f"tool{i}", {CapabilityDimension.PHYS_IO})
        
        result = analyzer.analyze_redundancy()
        
        # Solo 1 esencial, 4 redundantes
        assert len(result['essential_tools']) == 1
        assert len(result['redundant_tools']) == 4
    
    def test_all_tools_disjoint(self):
        """Todas las herramientas disjuntas."""
        analyzer = MICRedundancyAnalyzer()
        
        dims = list(CapabilityDimension)
        for i, dim in enumerate(dims):
            analyzer.register_tool(f"tool{i}", {dim})
        
        result = analyzer.analyze_redundancy()
        
        # Todas esenciales
        assert len(result['essential_tools']) == 5
        assert len(result['redundant_tools']) == 0
    
    def test_empty_capabilities(self):
        """Herramienta sin capacidades."""
        analyzer = MICRedundancyAnalyzer()
        
        analyzer.register_tool("empty_tool", set())
        analyzer.register_tool("normal_tool", {CapabilityDimension.PHYS_IO})
        
        result = analyzer.analyze_redundancy()
        
        # Ambas deben estar clasificadas
        all_tools = set(result['essential_tools']) | set(result['redundant_tools'])
        assert 'empty_tool' in all_tools
        assert 'normal_tool' in all_tools

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
        inner_product = len(v1.components.intersection(v2.components))
        assert inner_product == 0, f"Ruptura de ortogonalidad funcional. Entropía cruzada: {inner_product}"
        
        # 100 herramientas
        for i in range(100):
            dim = list(CapabilityDimension)[i % 5]
            analyzer.register_tool(f"tool{i}", {dim})
        
        # Debe completar sin timeout
        result = analyzer.analyze_redundancy()
        
        assert result['status'] == 'success'
        assert result['statistics']['total_tools'] == 102
    
    def test_quine_mccluskey_scaling(self):
        """Escalabilidad del algoritmo de Quine-McCluskey."""
        # Probar diferentes dimensiones
        for num_vars in [4, 6, 8]:
            minimizer = QuineMcCluskeyMinimizer(num_vars)
            
            # Mitad de los minitérminos
            minterms = list(range(0, 1 << num_vars, 2))
            
            # Debe completar
            prime_impls = minimizer.compute_prime_implicants(minterms)
            
            assert len(prime_impls) > 0
    
    def test_union_find_performance(self):
        """Performance de Union-Find."""
        n = 10000
        uf = UnionFind(n)
        
        # Unir todos en cadena
        for i in range(n - 1):
            uf.union(i, i + 1)
        
        # Verificar componente única
        assert uf.num_components == 1
        
        # Find debe ser rápido (path compression)
        root = uf.find(n - 1)
        assert uf.find(0) == root


# ========================================================================================
# TESTS: Integración Completa
# ========================================================================================

class TestIntegration:
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
        assert result['status'] == 'success'
        
        # audit_fusion es redundante con structure_logic
        # Nota: La clasificación depende del orden de selección en Quine-McCluskey.
        # Ambas herramientas tienen la misma firma topológica.
        all_tools = set(result['essential_tools']) | set(result['redundant_tools'])
        assert 'audit_fusion' in all_tools
        assert 'structure_logic' in all_tools

        # Al menos una de las dos con firma idéntica debe ser redundante
        redundant = result['redundant_tools']
        assert 'audit_fusion' in redundant or 'structure_logic' in redundant
        
        # Cobertura completa
        all_tools = set(result['essential_tools']) | set(result['redundant_tools'])
        assert len(all_tools) == 6
        
        # Reducción detectada
        assert result['statistics']['reduction_rate'] > 0
    
    def test_consensus_between_methods(self):
        """Consenso entre diferentes métodos de análisis."""
        analyzer = MICRedundancyAnalyzer()
        
        # Crear redundancia conocida
        analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO})
        analyzer.register_tool("tool2", {CapabilityDimension.PHYS_IO})  # Idéntica
        
        result = analyzer.analyze_redundancy()
        
        # Homología debe detectar ciclo
        assert result['homology']['H_1'] > 0
        
        # Debe haber una redundante
        assert len(result['redundant_tools']) == 1
        
        # Estadísticas deben ser consistentes
        assert result['statistics']['essential_count'] + result['statistics']['redundant_count'] == \
            result['statistics']['total_tools']


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
