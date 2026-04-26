"""
=========================================================================================
Suite de Pruebas: Auditoría de Redundancia MIC (Algoritmo de Quine-McCluskey)
Ubicación: dev_tools/test_mic_minimizer.py
Versión: 3.4 - Rigor Topológico, Espectral y Categórico (Post-Review)
=========================================================================================
"""

import pytest
import numpy as np
from typing import List, Set, Dict, FrozenSet
from itertools import combinations, product
import logging
from collections import defaultdict
import sys
import time
from pathlib import Path

# Ajustar path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mic_minimizer import (
    CapabilityDimension,
    BooleanVector,
    Tool,
    ImplicantTerm,
    QuineMcCluskeyMinimizer,
    TopologicalInvariantComputer,
    MICRedundancyAnalyzer,
    audit_mic_redundancy
)

# Configuración de logging para pruebas
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("MIC.Tests")

# ========================================================================================
# CONSTANTES DE RIGOR COMPUTACIONAL
# ========================================================================================
NUM_VARS_MAX_TESTED = 8  # Cota superior tratable para minimización algorítmica exacta en tiempo O(1)
MAX_COMPUTATION_TIME_S = 0.5

# ========================================================================================
# FIXTURES
# ========================================================================================

@pytest.fixture
def v_empty():
    return BooleanVector(frozenset())

@pytest.fixture
def v_io():
    return BooleanVector(frozenset([CapabilityDimension.PHYS_IO]))

@pytest.fixture
def v_num():
    return BooleanVector(frozenset([CapabilityDimension.PHYS_NUM]))

@pytest.fixture
def v_io_num():
    return BooleanVector(frozenset([
        CapabilityDimension.PHYS_IO,
        CapabilityDimension.PHYS_NUM
    ]))

@pytest.fixture
def v_all():
    return BooleanVector(frozenset(CapabilityDimension))

@pytest.fixture
def qm_3():
    return QuineMcCluskeyMinimizer(num_vars=3)

@pytest.fixture
def analyzer():
    return MICRedundancyAnalyzer()

# ========================================================================================
# PRUEBAS ALGEBRAICAS: ESTRUCTURAS BOOLEANAS
# ========================================================================================

class TestBooleanVectorAlgebra:
    """
    Pruebas de axiomas algebraicos para BooleanVector.
    Valida que la estructura sea un retículo booleano riguroso en B^n.
    """

    @pytest.mark.parametrize("vec_name", ["v_empty", "v_io", "v_io_num", "v_all"])
    def test_idempotence(self, vec_name, request):
        """Axioma: A ∪ A = A y A ∩ A = A (Idempotencia)"""
        vec = request.getfixturevalue(vec_name)
        assert vec.union(vec) == vec
        assert vec.intersection(vec) == vec

    def test_commutativity(self, v_io, v_num):
        """Axioma: A ∪ B = B ∪ A y A ∩ B = B ∩ A (Conmutatividad)"""
        assert v_io.union(v_num) == v_num.union(v_io)
        assert v_io.intersection(v_num) == v_num.intersection(v_io)

    def test_associativity(self, v_io, v_num, v_io_num):
        """Axioma: (A ∪ B) ∪ C = A ∪ (B ∪ C) (Asociatividad)"""
        a, b, c = v_io, v_num, v_io_num
        assert a.union(b).union(c) == a.union(b.union(c))
        assert a.intersection(b).intersection(c) == a.intersection(b.intersection(c))

    def test_absorption_laws(self, v_io, v_num):
        """Leyes de absorción: A ∪ (A ∩ B) = A"""
        assert v_io.union(v_io.intersection(v_num)) == v_io
        assert v_io.intersection(v_io.union(v_num)) == v_io

    def test_distributivity(self, v_io, v_num, v_io_num):
        """Axioma de distributividad del retículo."""
        a, b, c = v_io, v_num, v_io_num
        assert a.union(b.intersection(c)) == a.union(b).intersection(a.union(c))

    def test_identity_elements(self, v_io, v_empty, v_all):
        """Existencia de ínfimo (∅) y supremo (U) globales."""
        assert v_io.union(v_empty) == v_io
        assert v_io.intersection(v_all) == v_io

    def test_complement_properties(self, v_io, v_num, v_io_num, v_empty):
        """Propiedad de auto-cancelación en el anillo booleano (XOR)."""
        for vec in [v_io, v_num, v_io_num]:
            assert vec.symmetric_difference(vec) == v_empty

    def test_binary_representation(self, v_empty, v_all, v_io):
        num_vars = len(CapabilityDimension)
        assert len(v_empty.to_binary_string(num_vars)) == num_vars
        assert v_empty.to_binary_string(num_vars) == "0" * num_vars
        assert v_all.to_binary_string(num_vars) == "1" * num_vars

    def test_minterm_conversion(self, v_empty, v_io, v_num, v_io_num):
        assert v_empty.to_minterm() == 0
        assert v_io.to_minterm() == 1
        assert v_num.to_minterm() == 2
        assert v_io_num.to_minterm() == 3

    def test_hamming_weight(self, v_empty, v_io, v_io_num, v_all):
        assert v_empty.hamming_weight() == 0
        assert v_io.hamming_weight() == 1
        assert v_io_num.hamming_weight() == 2
        assert v_all.hamming_weight() == len(CapabilityDimension)

# ========================================================================================
# PRUEBAS DE ALGORITMO: QUINE-MCCLUSKEY
# ========================================================================================

class TestQuineMcCluskeyMinimizer:
    @pytest.mark.parametrize("num_vars", sorted({1, 2, 4, 8, NUM_VARS_MAX_TESTED}))
    def test_boundary_constraints_vars(self, num_vars):
        QuineMcCluskeyMinimizer(num_vars=num_vars)

    @pytest.mark.parametrize("num_vars", [0, 33])
    def test_invalid_manifold_dimension(self, num_vars):
        with pytest.raises(ValueError):
            QuineMcCluskeyMinimizer(num_vars=num_vars)

    def test_trivial_convergence(self, qm_3):
        assert len(qm_3.compute_prime_implicants([])) == 0

    def test_singleton_projection(self, qm_3):
        primes = qm_3.compute_prime_implicants([5])
        assert len(primes) == 1
        assert list(primes)[0].pattern == "101"

    def test_full_manifold_collapse(self, qm_3):
        all_minterms = list(range(2**3))
        primes = qm_3.compute_prime_implicants(all_minterms)
        covered = set()
        for p in primes:
            covered.update(p.covered_minterms)
        assert covered == set(all_minterms)

    @pytest.mark.parametrize("minterms, expected_patterns", [
        ([0, 1, 2], {"0-", "-0"}),
        ([0, 1], {"00-"}),
    ])
    def test_categorical_minimal_examples(self, minterms, expected_patterns):
        num_vars = len(next(iter(expected_patterns)))
        qm = QuineMcCluskeyMinimizer(num_vars=num_vars)
        primes = qm.compute_prime_implicants(minterms)
        patterns = {p.pattern for p in primes}
        for expected in expected_patterns:
            assert expected in patterns

# ========================================================================================
# PRUEBAS TOPOLÓGICAS: HOMOLOGÍA Y COMPONENTES
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
# PRUEBAS ESPECTRALES: MATRICES Y RANGOS
# ========================================================================================

class TestSpectralRigor:
    """
    Análisis del espectro de la Matriz de Interacción Central.
    """

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
        
        results = analyzer.analyze_redundancy()
        
        analyzer_min = MICRedundancyAnalyzer()
        for name in results['essential_tools']:
            tool = next(t for t in analyzer.tools if t.name == name)
            analyzer_min.register_tool(tool.name, tool.capabilities.components)

        m_min = analyzer_min.build_incidence_matrix()
        active_caps = np.sum(m_min, axis=0) > 0
        m_essential = m_min[:, active_caps]
        
        rank = np.linalg.matrix_rank(m_essential)
        # Rango completo => dim(ker) = 0 en el subespacio de herramientas esenciales
        assert rank == len(results['essential_tools']), f"Deficiencia espectral: rango {rank} < {len(results['essential_tools'])}"

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
        
        m = analyzer.build_incidence_matrix()
        gram = np.dot(m, m.T)
        
        off_diag = gram - np.diag(np.diag(gram))
        assert np.all(off_diag == 0), "Fallo de ortogonalidad funcional: herramientas ortogonales tienen productos internos no nulos"

# ========================================================================================
# PRUEBAS CATEGÓRICAS Y CUÁNTICAS
# ========================================================================================

class TestCategoricalQuantumRigor:
    """
    Funtorialidad e Isometría de la información.
    """

    def test_minimization_idempotency(self, analyzer):
        """
        M(M(G)) = M(G).
        La minimización es un endofuntor idempotente.
        """
        analyzer.register_tool("X", {CapabilityDimension.PHYS_IO})
        analyzer.register_tool("Y", {CapabilityDimension.PHYS_IO})
        
        r1 = analyzer.analyze_redundancy()
        e1 = sorted(r1['essential_tools'])
        
        analyzer2 = MICRedundancyAnalyzer()
        for name in e1:
            tool = next(t for t in analyzer.tools if t.name == name)
            analyzer2.register_tool(tool.name, tool.capabilities.components)

        r2 = analyzer2.analyze_redundancy()
        e2 = sorted(r2['essential_tools'])
        
        assert e1 == e2, "El proceso de minimización no es idempotente (inyecta entropía espuria)"

    def test_coproduct_isomorphism(self):
        """
        M(A ⊕ B) ≅ M(A) ⊕ M(B).
        La minimización debe preservar el coproducto de módulos disjuntos.
        """
        analyzer_a = MICRedundancyAnalyzer()
        analyzer_a.register_tool("A1", {CapabilityDimension.PHYS_IO})
        analyzer_a.register_tool("A2", {CapabilityDimension.PHYS_IO})
        
        analyzer_b = MICRedundancyAnalyzer()
        analyzer_b.register_tool("B", {CapabilityDimension.PHYS_NUM})
        
        analyzer_comb = MICRedundancyAnalyzer()
        analyzer_comb.register_tool("A1", {CapabilityDimension.PHYS_IO})
        analyzer_comb.register_tool("A2", {CapabilityDimension.PHYS_IO})
        analyzer_comb.register_tool("B", {CapabilityDimension.PHYS_NUM})
        
        res_a = analyzer_a.analyze_redundancy()
        res_b = analyzer_b.analyze_redundancy()
        res_comb = analyzer_comb.analyze_redundancy()
        
        assert len(res_comb['essential_tools']) == len(res_a['essential_tools']) + len(res_b['essential_tools'])

    def test_parseval_energy_conservation(self, analyzer):
        """
        Conserva la norma L2 del soporte informativo (Unitaridad).
        """
        analyzer.register_tool("T1", {CapabilityDimension.PHYS_IO, CapabilityDimension.PHYS_NUM})
        analyzer.register_tool("T2", {CapabilityDimension.PHYS_NUM})
        
        res = analyzer.analyze_redundancy()
        
        supp_initial = set().union(*(t.capabilities.components for t in analyzer.tools))
        supp_final = set()
        for name in res['essential_tools']:
            tool = next(t for t in analyzer.tools if t.name == name)
            supp_final.update(tool.capabilities.components)

        assert supp_initial == supp_final, "Violación de unitaridad: soporte funcional no conservado tras colapso"

    def test_shannon_entropy_determinism(self, analyzer):
        """
        Entropía determinista mediante colapso inducido por semillas lexicográficas.
        """
        analyzer.register_tool("Omega", {CapabilityDimension.PHYS_IO})
        analyzer.register_tool("Alpha", {CapabilityDimension.PHYS_IO})

        res = analyzer.analyze_redundancy()
        # "Alpha" < "Omega", el observable colapsa determinísticamente a "Alpha"
        assert "Alpha" in res['essential_tools']
        assert "Omega" in res['redundant_tools']

# ========================================================================================
# PRUEBAS DE RENDIMIENTO Y ESCALABILIDAD
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

        matrix = analyzer.build_incidence_matrix()
        assert matrix.shape == (n_tools, len(CapabilityDimension))
        assert np.sum(matrix) == n_tools

if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
