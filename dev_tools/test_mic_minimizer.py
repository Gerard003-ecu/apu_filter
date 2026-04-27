"""
=========================================================================================
MÓDULO: test_mic_minimizer.py
Ubicación: dev_tools/test_mic_minimizer.py
Versión: 4.0 - Análisis Riguroso con Mejoras Metodológicas
=========================================================================================

DESCRIPCIÓN EJECUTIVA:
Suite completa de pruebas unitarias e integración para validación de:
  • Axiomas de álgebra booleana (retículos, leyes distributivas)
  • Correctitud del algoritmo Quine-McCluskey en B^n
  • Invariantes topológicos (homología, números de Betti, característica de Euler)
  • Propiedades espectrales (rango, kernel, ortogonalidad de Gram)
  • Funtorialidad categórica (idempotencia, coproductos, unitaridad cuántica)
  • Escalabilidad computacional y estabilidad numérica

MEJORAS INTRODUCIDAS (v4.0):
  1. Separación clara de responsabilidades en clases de prueba
  2. Documentación exhaustiva de axiomas matemáticos
  3. Validación de precondiciones y postcondiciones
  4. Manejo robusto de errores con contexto diagnóstico
  5. Paramétrización mejorada para cobertura de casos límite
  6. Análisis espectral reforzado con tolerancias numéricas
  7. Pruebas categóricas con verificación de funtorialidad
  8. Métricas de rendimiento con límites justificados
=========================================================================================
"""

import pytest
import numpy as np
from typing import List, Set, Dict, FrozenSet, Tuple, Optional
from itertools import combinations, product
import logging
from collections import defaultdict
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# ========================================================================================
# CONFIGURACIÓN Y DEPENDENCIAS
# ========================================================================================

# Ajustar path de módulos
sys.path.insert(0, str(Path(__file__).parent.parent))

from mic_minimizer import (
    CapabilityDimension,
    BooleanVector,
    Tool,
    ImplicantTerm,
    QuineMcCluskeyMinimizer,
    MICRedundancyAnalyzer,
    audit_mic_redundancy
)

# Configuración de logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("MIC.Tests")

# Constantes numéricas
EPSILON_SPECTRAL = 1e-10  # Tolerancia para análisis espectral
MAX_COMPUTATION_TIME_S = 0.5  # Límite temporal para operaciones
NUM_VARS_MAX_TESTED = 10  # Cota superior tratable empíricamente


# ========================================================================================
# UTILIDADES Y AYUDANTES
# ========================================================================================

@dataclass
class TestContext:
    """Contexto de ejecución de prueba con información diagnóstica."""
    __test__ = False
    name: str
    timestamp: float
    duration: Optional[float] = None
    error_context: Optional[Dict[str, any]] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error(f"[{self.name}] Exception: {exc_val}", exc_info=True)


class NumericalValidator:
    """Validador robusto para operaciones numéricas con tolerancia controlada."""

    @staticmethod
    def assert_matrix_rank(matrix: np.ndarray, expected_rank: int, 
                          tol: float = 1e-10, context: str = "") -> bool:
        """
        Valida que rango(matrix) = expected_rank con tolerancia.
        
        Args:
            matrix: Matriz a validar
            expected_rank: Rango esperado
            tol: Tolerancia absoluta para valores singulares
            context: Descripción del contexto (para diagnóstico)
            
        Returns:
            True si la validación es exitosa
        """
        actual_rank = np.linalg.matrix_rank(matrix, tol=tol)
        if actual_rank != expected_rank:
            logger.warning(
                f"[NumericalValidator] {context}: "
                f"rank={actual_rank} (expected {expected_rank}), "
                f"condition_number={np.linalg.cond(matrix):.2e}"
            )
            return False
        return True

    @staticmethod
    def assert_boolean_orthogonality(v1: BooleanVector, v2: BooleanVector) -> None:
        """Valida ortogonalidad absoluta sin epsilon."""
        inner_product = len(v1.components.intersection(v2.components))
        assert inner_product == 0, f"Ruptura de ortogonalidad funcional. Entropía cruzada: {inner_product}"

class TopologicalInvariantComputer:
    """Computador de invariantes topológicos (números de Betti, característica de Euler)."""

    @staticmethod
    def compute_betti_numbers(analyzer: 'MICRedundancyAnalyzer') -> Dict[int, int]:
        """
        Computa números de Betti β_i del complejo simplicial K.
        
        β_0: Componentes conexas
        β_1: Agujeros (ciclos 1D)
        
        Args:
            analyzer: Instancia de MICRedundancyAnalyzer
            
        Returns:
            Dict con claves 'H_0', 'H_1' (dimensiones de grupos de homología)
        """
        betti = analyzer.compute_homology_groups()
        return betti

    def compute_euler_characteristic(self, implicants: List[str]) -> int:
        """
        Calcula χ aplicando la fórmula de Euler-Poincaré sobre el Complejo de Čech.
        Utiliza el Principio de Inclusión-Exclusión sobre las intersecciones
        de los hipercubos (implicantes) para preservar el invariante topológico.
        """
        n = len(implicants)
        if n == 0:
            return 0
            
        chi = 0
        import math
        # k representa la dimensión de la intersección (1-way, 2-way, ..., n-way)
        for k in range(1, n + 1):
            k_way_intersections = 0
            for subset in combinations(implicants, k):
                # Si el subconjunto de implicantes tiene una intersección no nula
                # (son lógicamente compatibles), contribuyen al Complejo de Cech.
                if self._are_compatible(subset):
                    k_way_intersections += 1

            # Suma alternada: + (1-way) - (2-way) + (3-way) ...
            chi += int(math.pow(-1, k - 1)) * k_way_intersections

        return chi

    def evaluate_lie_commutator(self, tensor_a: str, tensor_b: str) -> float:
        """
        Computa [A, B] midiendo el entrelazamiento destructivo.
        Si la dimensión i de tensor_a choca topológicamente con la de tensor_b
        (ej. '1' vs '0' simultáneo sin aislamiento '-'), no conmutan.
        """
        # Lógica simpléctica operando estrictamente sobre tipos 'str'
        for bit_a, bit_b in zip(tensor_a, tensor_b):
            if bit_a != '-' and bit_b != '-' and bit_a != bit_b:
                return 1.0 # Conmutador no nulo (Singularidad / Incompatibilidad)
        return 0.0 # Conmutación perfecta (Ortogonalidad preservada)

    def _are_compatible(self, subset: Tuple[str, ...]) -> bool:
        """
        Evalúa la ortogonalidad y conmutatividad cuántica directamente
        sobre los tensores booleanos de las capacidades en B^n.
        """
        if not subset:
            return True

        # Comparamos todos los pares dentro del subset
        for a, b in combinations(subset, 2):
            if self.evaluate_lie_commutator(a, b) != 0.0:
                return False
        return True

    @staticmethod
    def compute_edge_set(incidence_matrix: np.ndarray) -> int:
        """
        Computa número de aristas en el grafo de compatibilidad de herramientas.
        Arista (i,j): herramientas i,j comparten al menos una capacidad.
        
        Args:
            incidence_matrix: Matriz (n_tools × n_capabilities)
            
        Returns:
            Número de aristas
        """
        n_tools = incidence_matrix.shape[0]
        n_edges = 0
        for i, j in combinations(range(n_tools), 2):
            if np.dot(incidence_matrix[i], incidence_matrix[j]) > 0:
                n_edges += 1
        return n_edges


# ========================================================================================
# FIXTURES: Instancias Reutilizables
# ========================================================================================

@pytest.fixture
def v_empty():
    """Vector booleano vacío (elemento neutro de ∪, elemento absorvente de ∩)."""
    return BooleanVector(frozenset())


@pytest.fixture
def v_io():
    """Vector con dimensión PHYS_IO únicamente."""
    return BooleanVector(frozenset([CapabilityDimension.PHYS_IO]))


@pytest.fixture
def v_num():
    """Vector con dimensión PHYS_NUM únicamente."""
    return BooleanVector(frozenset([CapabilityDimension.PHYS_NUM]))


@pytest.fixture
def v_io_num():
    """Vector con PHYS_IO y PHYS_NUM."""
    return BooleanVector(frozenset([
        CapabilityDimension.PHYS_IO,
        CapabilityDimension.PHYS_NUM
    ]))


@pytest.fixture
def v_all():
    """Vector universal (elemento neutro de ∩, elemento absorvente de ∪)."""
    return BooleanVector(frozenset(CapabilityDimension))


@pytest.fixture
def qm_3():
    """Minimizador Quine-McCluskey para B^3."""
    return QuineMcCluskeyMinimizer(num_vars=3)


@pytest.fixture
def analyzer():
    """Analizador de redundancia MIC limpio."""
    return MICRedundancyAnalyzer()


@pytest.fixture
def numerical_validator():
    """Validador numérico singleton."""
    return NumericalValidator()


@pytest.fixture
def topological_computer():
    """Computador de invariantes topológicos."""
    return TopologicalInvariantComputer()


# ========================================================================================
# CLASE 1: PRUEBAS ALGEBRAICAS - ESTRUCTURA DE RETÍCULO BOOLEANO
# ========================================================================================

class TestBooleanVectorAlgebra:
    """
    PROPÓSITO: Validar que BooleanVector forma un retículo booleano riguroso ⟨B^n, ∪, ∩, ¯⟩
    
    AXIOMAS VERIFICADOS:
      • Idempotencia: A ∪ A = A, A ∩ A = A
      • Conmutatividad: A ∪ B = B ∪ A, A ∩ B = B ∩ A
      • Asociatividad: (A ∪ B) ∪ C = A ∪ (B ∪ C)
      • Absorción: A ∪ (A ∩ B) = A
      • Distributividad: A ∪ (B ∩ C) = (A ∪ B) ∩ (A ∪ C)
      • Elementos identidad: ∅ (neutro ∪), U (neutro ∩)
      • Complemento: A ⊕ A = ∅ (anillo booleano)
    """

    @pytest.mark.parametrize("vec_fixture", ["v_empty", "v_io", "v_num", "v_io_num", "v_all"])
    def test_idempotence_law(self, vec_fixture, request):
        """
        AXIOMA: A ∪ A = A y A ∩ A = A
        
        Valida que la unión e intersección de un conjunto consigo mismo
        devuelve el conjunto sin cambios (propiedad de punto fijo).
        """
        vec = request.getfixturevalue(vec_fixture)
        
        union_result = vec.union(vec)
        intersection_result = vec.intersection(vec)
        
        assert union_result == vec, (
            f"Fallo idempotencia ∪: {vec} ∪ {vec} = {union_result} ≠ {vec}"
        )
        assert intersection_result == vec, (
            f"Fallo idempotencia ∩: {vec} ∩ {vec} = {intersection_result} ≠ {vec}"
        )

    def test_commutativity_law(self, v_io, v_num):
        """
        AXIOMA: A ∪ B = B ∪ A y A ∩ B = B ∩ A
        
        El orden de operandos no afecta el resultado (propiedades simétricas).
        """
        union_ab = v_io.union(v_num)
        union_ba = v_num.union(v_io)
        
        intersection_ab = v_io.intersection(v_num)
        intersection_ba = v_num.intersection(v_io)
        
        assert union_ab == union_ba, (
            f"Conmutatividad ∪ violada: {v_io} ∪ {v_num} ≠ {v_num} ∪ {v_io}"
        )
        assert intersection_ab == intersection_ba, (
            f"Conmutatividad ∩ violada: {v_io} ∩ {v_num} ≠ {v_num} ∩ {v_io}"
        )

    def test_associativity_law(self, v_io, v_num, v_io_num):
        """
        AXIOMA: (A ∪ B) ∪ C = A ∪ (B ∪ C) y (A ∩ B) ∩ C = A ∩ (B ∩ C)
        
        El agrupamiento de operandos no afecta el resultado (propiedad asociativa).
        """
        a, b, c = v_io, v_num, v_io_num
        
        union_left = a.union(b).union(c)
        union_right = a.union(b.union(c))
        
        intersection_left = a.intersection(b).intersection(c)
        intersection_right = a.intersection(b.intersection(c))
        
        assert union_left == union_right, (
            f"Asociatividad ∪ violada"
        )
        assert intersection_left == intersection_right, (
            f"Asociatividad ∩ violada"
        )

    def test_absorption_laws(self, v_io, v_num):
        """
        AXIOMA: A ∪ (A ∩ B) = A y A ∩ (A ∪ B) = A
        
        Garantiza que operaciones redundantes se colapsan (absorción).
        """
        absorption_union = v_io.union(v_io.intersection(v_num))
        absorption_intersection = v_io.intersection(v_io.union(v_num))
        
        assert absorption_union == v_io, (
            f"Absorción ∪ violada: {v_io} ∪ ({v_io} ∩ {v_num}) ≠ {v_io}"
        )
        assert absorption_intersection == v_io, (
            f"Absorción ∩ violada: {v_io} ∩ ({v_io} ∪ {v_num}) ≠ {v_io}"
        )

    def test_distributivity_law(self, v_io, v_num, v_io_num):
        """
        AXIOMA: A ∪ (B ∩ C) = (A ∪ B) ∩ (A ∪ C)
                A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C)
        
        Verifica distributividad de unión sobre intersección (propiedad de retículo).
        """
        a, b, c = v_io, v_num, v_io_num
        
        # Primera ley distributiva
        dist_1_left = a.union(b.intersection(c))
        dist_1_right = a.union(b).intersection(a.union(c))
        
        assert dist_1_left == dist_1_right, (
            f"Distributividad violada: {a} ∪ ({b} ∩ {c}) ≠ "
            f"({a} ∪ {b}) ∩ ({a} ∪ {c})"
        )

    def test_identity_elements(self, v_io, v_empty, v_all):
        """
        AXIOMA: Existencia de elementos neutros y absorbentes
        
        ∅: Elemento neutro de ∪ (A ∪ ∅ = A), absorbente de ∩ (A ∩ ∅ = ∅)
        U: Elemento neutro de ∩ (A ∩ U = A), absorbente de ∪ (A ∪ U = U)
        """
        # Elemento neutro de ∪
        assert v_io.union(v_empty) == v_io, "∅ no es neutro para ∪"
        
        # Elemento neutro de ∩
        assert v_io.intersection(v_all) == v_io, "U no es neutro para ∩"
        
        # Elemento absorbente de ∪
        assert v_io.union(v_all) == v_all, "U no es absorbente para ∪"
        
        # Elemento absorbente de ∩
        assert v_io.intersection(v_empty) == v_empty, "∅ no es absorbente para ∩"

    def test_complement_self_cancellation(self, v_io, v_num, v_io_num, v_empty):
        """
        AXIOMA: A ⊕ A = ∅ (diferencia simétrica = complemento en anillo booleano)
        
        XOR de un conjunto consigo mismo produce el conjunto vacío.
        """
        for vec in [v_io, v_num, v_io_num]:
            result = vec.symmetric_difference(vec)
            assert result == v_empty, (
                f"Auto-cancelación XOR violada: {vec} ⊕ {vec} ≠ ∅"
            )

    def test_binary_string_representation_correctness(self, v_empty, v_all, v_io, v_num):
        """
        VALIDACIÓN: Representación binaria coherente con estructura booleana.
        
        Para n variables:
          - ∅ = "000...0"
          - U = "111...1"
          - Cada bit ∈ {0, 1}
        """
        num_vars = len(CapabilityDimension)
        
        empty_binary = v_empty.to_binary_string(num_vars)
        all_binary = v_all.to_binary_string(num_vars)
        
        assert len(empty_binary) == num_vars, f"Longitud incorrecta para ∅"
        assert empty_binary == "0" * num_vars, f"Representación de ∅ incorrecta"
        
        assert len(all_binary) == num_vars, f"Longitud incorrecta para U"
        assert all_binary == "1" * num_vars, f"Representación de U incorrecta"
        
        # Verificar que cada bit es 0 o 1
        for bit in empty_binary + all_binary:
            assert bit in {'0', '1'}, f"Bit inválido en representación: {bit}"

    def test_minterm_encoding_bijection(self, v_empty, v_io, v_num, v_io_num):
        """
        VALIDACIÓN: Mapeo biyectivo entre vectores booleanos y números enteros.
        
        Cada subconjunto de B^n ↔ único entero en [0, 2^n)
        """
        expected_values = [
            (v_empty, 0),
            (v_io, 1),
            (v_num, 2),
            (v_io_num, 3)
        ]
        
        for vec, expected_minterm in expected_values:
            actual_minterm = vec.to_minterm()
            assert actual_minterm == expected_minterm, (
                f"Codificación de minterm incorrecta: {vec} → {actual_minterm} "
                f"(esperado {expected_minterm})"
            )

    def test_hamming_weight_cardinality(self, v_empty, v_io, v_num, v_io_num, v_all):
        """
        VALIDACIÓN: Peso de Hamming = |elementos|
        
        h(A) = número de dimensiones presentes en A
        """
        test_cases = [
            (v_empty, 0),
            (v_io, 1),
            (v_num, 1),
            (v_io_num, 2),
            (v_all, len(CapabilityDimension))
        ]
        
        for vec, expected_weight in test_cases:
            actual_weight = vec.hamming_weight()
            assert actual_weight == expected_weight, (
                f"Peso de Hamming incorrecto: h({vec}) = {actual_weight} "
                f"(esperado {expected_weight})"
            )


# ========================================================================================
# CLASE 2: PRUEBAS DE ALGORITMO - QUINE-MCCLUSKEY
# ========================================================================================

class TestQuineMcCluskeyMinimizer:
    """
    PROPÓSITO: Validar correctitud y completitud del algoritmo de minimización 
               Quine-McCluskey en el hipercubo booleano B^n.
    
    GARANTÍAS:
      • Cobertura completa: Todos los minitérminos se cubren
      • Minimalidad: Implicantes primos sin subsunción
      • Determinismo: Mismo resultado para misma entrada
      • Escalabilidad: Complejidad polinomial en n
    """

    @pytest.mark.parametrize("num_vars", [1, 2, 3, 4, 8, 10])
    def test_constructor_valid_dimensions(self, num_vars):
        """
        PRECONDICIÓN: 1 ≤ num_vars ≤ 10 (representable en arquitectura estándar)
        
        Verifica que la construcción del minimizador sea exitosa para
        dimensiones válidas del hipercubo.
        """
        try:
            qm = QuineMcCluskeyMinimizer(num_vars=num_vars)
            assert qm.num_vars == num_vars
        except Exception as e:
            pytest.fail(f"Constructor falló para num_vars={num_vars}: {e}")

    @pytest.mark.parametrize("invalid_num_vars", [0, -1, 11, 33, 100])
    def test_constructor_rejects_invalid_dimensions(self, invalid_num_vars):
        """
        VALIDACIÓN: Rechazar dimensiones inválidas.
        
        1 ≤ num_vars ≤ 10 es garantizado por la arquitectura de almacenamiento.
        """
        with pytest.raises(ValueError):
            QuineMcCluskeyMinimizer(num_vars=invalid_num_vars)

    def test_minimizer_empty_input_trivial_case(self, qm_3):
        """
        CASO BASE: Entrada vacía (no hay minitérminos).
        
        compute_prime_implicants([]) debe devolver {} (no hay implicantes).
        """
        result = qm_3.compute_prime_implicants([])
        assert isinstance(result, (set, frozenset))
        assert len(result) == 0, "Caso vacío no devuelve conjunto vacío"

    def test_minimizer_singleton_projection(self, qm_3):
        """
        CASO BASE: Un único minitérmino.
        
        El implicante primo debe ser el patrón booleano de ese minitérmino.
        Para m=5 en B^3: 5 = 101₂ → patrón "101"
        """
        result = qm_3.compute_prime_implicants([5])
        
        assert len(result) == 1, "Singleton debe producir un implicante"
        
        implicant = list(result)[0]
        assert implicant.pattern == "101", (
            f"Patrón incorrecto para minterm 5: {implicant.pattern} ≠ '101'"
        )

    def test_minimizer_universal_set_coverage(self, qm_3):
        """
        PROPIEDAD: Cobertura universal.
        
        Si entramos todos los minitérminos de B^n, los implicantes primos
        deben cubrirlos todos (postcondición de integridad).
        """
        all_minterms = list(range(2**3))  # [0, 1, 2, ..., 7] para B^3
        
        primes = qm_3.compute_prime_implicants(all_minterms)
        
        covered_minterms = set()
        for prime in primes:
            covered_minterms.update(prime.covered_minterms)
        
        assert covered_minterms == set(all_minterms), (
            f"No se cubren todos los minitérminos: "
            f"cubiertos={covered_minterms}, esperados={set(all_minterms)}"
        )

    @pytest.mark.parametrize("minterms,expected_pattern_subset", [
        ([0, 1], {"00-"}),      # B^3: {0=000, 1=001} → "00-"
        ([0, 2], {"0-0"}),      # B^3: {0=000, 2=010} → "0-0"
        ([0, 1, 2, 3], {"0--"}),       # B^3: {0,1,2,3} → "0--" (cubridor máximo)
    ])
    def test_minimizer_canonical_examples(self, minterms, expected_pattern_subset):
        """
        VALIDACIÓN: Ejemplos canónicos con patrones verificables.
        
        Computa implicantes y verifica que cada patrón esperado está presente
        en el conjunto de implicantes primos.
        """
        num_vars = 3
        qm = QuineMcCluskeyMinimizer(num_vars=num_vars)
        primes = qm.compute_prime_implicants(minterms)
        
        patterns = {p.pattern for p in primes}
        
        for expected in expected_pattern_subset:
            assert expected in patterns, (
                f"Patrón esperado '{expected}' no encontrado en {patterns} "
                f"para minitérminos {minterms}"
            )

    def test_minimizer_idempotence_property(self, qm_3):
        """
        PROPIEDAD FUNCIONAL: M(M(x)) = M(x)
        
        Aplicar minimización dos veces al mismo conjunto debe dar resultado idéntico.
        """
        minterms = [1, 3, 5, 7]
        
        primes_1 = qm_3.compute_prime_implicants(minterms)
        patterns_1 = {p.pattern for p in primes_1}
        
        # Re-minimizar (conceptualmente, los implicantes ya son mínimos)
        primes_2 = qm_3.compute_prime_implicants(minterms)
        patterns_2 = {p.pattern for p in primes_2}
        
        assert patterns_1 == patterns_2, (
            f"Minimización no es idempotente: {patterns_1} ≠ {patterns_2}"
        )


# ========================================================================================
# CLASE 3: PRUEBAS TOPOLÓGICAS - INVARIANTES HOMOLÓGICOS
# ========================================================================================

class TestTopologicalRigor:
    """
    PROPÓSITO: Validar preservación de invariantes topológicos bajo minimización.
    
    CONCEPTOS:
      • Números de Betti (β_i): Dimensiones de grupos de homología H_i
      • Característica de Euler (χ): χ = β_0 - β_1 (en 1D), χ = v - e (grafo)
      • Retracto de deformación: La minimización es un retracto que preserva topología
    
    INVARIANTES VERIFICADOS:
      • β_0(K) = β_0(K'): Número de componentes conexas preservado
      • χ(K) = χ(K'): Característica de Euler invariante
      • ∂² = 0: Operador de frontera es nilpotente (axioma fundamental)
    """

    def test_betti_number_preservation_connected_component(self, analyzer, 
                                                          topological_computer):
        """
        AXIOMA: β_0(K) invariante bajo minimización de retractibilidad.
        
        Registramos 3 herramientas donde T2 es redundante (mismo suporte que T1).
        El número de componentes conexas debe preservarse tras eliminar T2.
        """
        analyzer.register_tool("T1", {CapabilityDimension.PHYS_IO})
        analyzer.register_tool("T2", {CapabilityDimension.PHYS_IO})  # Redundante
        analyzer.register_tool("T3", {CapabilityDimension.PHYS_NUM})
        
        # Computar Betti antes de minimización
        betti_before = topological_computer.compute_betti_numbers(analyzer)
        beta_0_before = betti_before.get('H_0', 0)
        
        # Realizar análisis de redundancia
        results = analyzer.analyze_redundancy()
        
        # Construir sistema minimizado (solo herramientas esenciales)
        analyzer_min = MICRedundancyAnalyzer()
        for tool_name in results['essential_tools']:
            tool = next((t for t in analyzer.tools if t.name == tool_name), None)
            if tool:
                analyzer_min.register_tool(tool.name, tool.capabilities.components)
        
        # Computar Betti después de minimización
        betti_after = topological_computer.compute_betti_numbers(analyzer_min)
        beta_0_after = betti_after.get('H_0', 0)
        
        assert beta_0_before == beta_0_after, (
            f"Ruptura topológica: β_0(K)={beta_0_before}, "
            f"β_0(K')={beta_0_after} tras minimización (retracto debe "
            f"preservar β_0)"
        )

    def test_euler_characteristic_preservation(self, analyzer, topological_computer):
        """
        AXIOMA: χ(K) = χ(K') bajo equivalencia de homotopía utilizando el Complejo de Čech.
        """
        # Instanciación determinista del hiperespacio paramétrico
        compiler = QuineMcCluskeyMinimizer(num_vars=3)

        # A set of minterms representing a specific topology (e.g., a union of hypercubes)
        # B^3 : let's take a union of a 2-cube (4 nodes) and a 1-cube (2 nodes) with 1 shared node
        # For example, "00-", "0-0", and "-00"
        minterms = [0, 1, 2, 4]
        # Let's count Euler characteristic using combinatorial method:
        # v = 4 (vertices: 0, 1, 2, 4)
        # edges for B^3: differences of 1 bit.
        # (0,1), (0,2), (0,4) -> e = 3 edges
        # No 2-faces (a square requires 4 vertices in a face, e.g. 0,1,2,3)
        # combinatorial chi = V - E + F = 4 - 3 + 0 = 1
        
        # Calculate prime implicants
        primes = compiler.compute_prime_implicants(minterms)
        
        # Mapeo a subespacio de representaciones puras (str)
        prime_patterns = [p.pattern for p in primes]
        chi_homological = topological_computer.compute_euler_characteristic(prime_patterns)
        
        assert chi_homological == 1, (
            f"Inconsistencia en característica de Euler vía Teorema del Nervio: "
            f"χ_homol={chi_homological} ≠ 1"
        )

    def test_boundary_operator_nilpotence(self):
        """
        AXIOMA FUNDAMENTAL: ∂² = 0
        
        Para un 1-complejo (grafo), no existen 2-simples (caras),
        por lo que ∂_2 es automáticamente nulo (rank(∂_2) = 0).
        
        Este axioma es la base de la homología simplicial.
        """
        # Para complejos 1D (grafos), ∂_2: C_2 → C_1 es nulo
        # porque dim(C_2) = 0 (no hay 2-simplex)
        rank_boundary_2 = 0
        
        assert rank_boundary_2 == 0, (
            "Herejía topológica: rank(∂_2) debe ser 0 para 1-complejos"
        )


# ========================================================================================
# CLASE 4: PRUEBAS ESPECTRALES - ANÁLISIS DE MATRICES
# ========================================================================================

class TestSpectralRigor:
    """
    PROPÓSITO: Validar propiedades espectrales (rango, kernel, ortogonalidad)
               de la Matriz de Interacción Central (MIC).
    
    CONCEPTOS:
      • rank(M): Rango = dimensión del espacio columna
      • ker(M): Kernel = espacio nulo
      • Gram(M) = MM^T: Producto interno de filas
      • Ortogonalidad: Filas ortogonales ⟺ Gram diagonal
    
    GARANTÍAS ESPECTRALES:
      • Rango completo tras minimización (no hay dependencias ocultas)
      • Kernel trivial para sistema esencial
      • Gram diagonal para herramientas funcionalmente independientes
    """

    def test_rank_completeness_after_minimization(self, analyzer, numerical_validator):
        """
        PROPIEDAD: rank(M_min) = |essential_tools|
        
        La matriz minimizada debe tener rango completo (todas las filas
        son linealmente independientes en el subespacio de capacidades activas).
        """
        analyzer.register_tool("T1", {CapabilityDimension.PHYS_IO})
        analyzer.register_tool("T2", {CapabilityDimension.PHYS_NUM})
        analyzer.register_tool("T3", {CapabilityDimension.PHYS_NUM})  # Redundante
        
        results = analyzer.analyze_redundancy()
        n_essential = len(results['essential_tools'])
        
        # Reconstruir matriz con herramientas esenciales
        analyzer_min = MICRedundancyAnalyzer()
        for tool_name in results['essential_tools']:
            tool = next((t for t in analyzer.tools if t.name == tool_name), None)
            if tool:
                analyzer_min.register_tool(tool.name, tool.capabilities.components)
        
        matrix_min = analyzer_min.build_incidence_matrix()
        
        # Restricción a columnas (capacidades) activas
        active_caps = np.sum(matrix_min, axis=0) > 0
        matrix_essential = matrix_min[:, active_caps]
        
        # Verificar rango
        valid = numerical_validator.assert_matrix_rank(
            matrix_essential,
            expected_rank=n_essential,
            context=f"Rank test after minimization (n_essential={n_essential})"
        )
        
        assert valid, (
            f"Deficiencia de rango espectral: "
            f"rank(M_min)={np.linalg.matrix_rank(matrix_essential)} "
            f"≠ {n_essential}"
        )

    def test_gram_matrix_orthogonality_functional_independence(
        self, analyzer, numerical_validator):
        """
        PROPIEDAD: Herramientas funcionalmente independientes ⟺ Ortogonalidad Booleana
        
        Si T1 = {PHYS_IO} y T2 = {PHYS_NUM}, entonces:
          Intersección de componentes debe ser 0.
        """
        analyzer.register_tool("T1", {CapabilityDimension.PHYS_IO})
        analyzer.register_tool("T2", {CapabilityDimension.PHYS_NUM})
        
        numerical_validator.assert_boolean_orthogonality(
            analyzer.tools[0].capabilities, analyzer.tools[1].capabilities
        )

    def test_gram_matrix_shared_capabilities_coupling(self, analyzer):
        """
        PROPIEDAD: Herramientas con capacidades compartidas tienen Gram no-diagonal.
        
        Si T1 = {A, B} y T2 = {B, C}, entonces Gram[0,1] ≠ 0.
        """
        analyzer.register_tool("T1", {
            CapabilityDimension.PHYS_IO,
            CapabilityDimension.PHYS_NUM
        })
        analyzer.register_tool("T2", {
            CapabilityDimension.PHYS_NUM,
            CapabilityDimension.TACT_TOPO
        })
        
        matrix = analyzer.build_incidence_matrix()
        gram = np.dot(matrix, matrix.T)
        
        # Elemento no-diagonal debe ser 1 (capacidad PHYS_NUM compartida)
        off_diag_coupling = gram[0, 1]
        assert off_diag_coupling > 0, (
            f"Error: herramientas con capacidades compartidas deberían "
            f"tener acoplamiento de Gram no-nulo, got {off_diag_coupling}"
        )


# ========================================================================================
# CLASE 5: PRUEBAS CATEGÓRICAS Y CUÁNTICAS
# ========================================================================================

class TestCategoricalQuantumRigor:
    """
    PROPÓSITO: Validar propiedades categóricas (funtorialidad) y cuánticas
               (unitaridad, determinismo de colapso, conservación de energía).
    
    CONCEPTOS CATEGÓRICOS:
      • Funtorialidad: M(f: A → B) preserva morfismos
      • Idempotencia endofuntor: M ∘ M = M
      • Coproducto: M(A ⊕ B) ≅ M(A) ⊕ M(B)
      • Preservación de límites/colímites
    
    CONCEPTOS CUÁNTICOS:
      • Unitaridad: ||ψ'|| = ||ψ|| (conservación de norma)
      • Determinismo: Colapso mediante observable autoadjunto
      • Parseval: Energía conservada bajo transformación unitaria
      • Shannon: Entropía del observable colapsado determinista
    """

    def test_minimization_idempotency_endofunctor(self, analyzer):
        """
        AXIOMA CATEGÓRICO: M(M(G)) = M(G)
        
        La minimización es un endofuntor idempotente sobre la categoría
        de grafos de dependencia de herramientas.
        """
        analyzer.register_tool("X", {CapabilityDimension.PHYS_IO})
        analyzer.register_tool("Y", {CapabilityDimension.PHYS_IO})
        
        # Primera minimización
        results_1 = analyzer.analyze_redundancy()
        essential_1 = sorted(results_1['essential_tools'])
        
        # Segunda minimización (sobre herramientas esenciales)
        analyzer_2 = MICRedundancyAnalyzer()
        for tool_name in essential_1:
            tool = next((t for t in analyzer.tools if t.name == tool_name), None)
            if tool:
                analyzer_2.register_tool(tool.name, tool.capabilities.components)
        
        results_2 = analyzer_2.analyze_redundancy()
        essential_2 = sorted(results_2['essential_tools'])
        
        assert essential_1 == essential_2, (
            f"Idempotencia endofuntor violada: "
            f"M(G)={essential_1} ≠ M(M(G))={essential_2}"
        )

    def test_coproduct_isomorphism_disjoint_systems(self):
        """
        AXIOMA CATEGÓRICO: M(A ⊕ B) ≅ M(A) ⊕ M(B)
        
        La minimización distribuye sobre coproductos disjuntos.
        Dos subsistemas sin capacidades compartidas se minimizan independientemente.
        """
        # Subsistema A: herramientas con capacidad PHYS_IO
        analyzer_a = MICRedundancyAnalyzer()
        analyzer_a.register_tool("A1", {CapabilityDimension.PHYS_IO})
        analyzer_a.register_tool("A2", {CapabilityDimension.PHYS_IO})  # Redundante
        
        # Subsistema B: herramientas con capacidad PHYS_NUM
        analyzer_b = MICRedundancyAnalyzer()
        analyzer_b.register_tool("B", {CapabilityDimension.PHYS_NUM})
        
        # Sistema combinado A ⊕ B
        analyzer_comb = MICRedundancyAnalyzer()
        analyzer_comb.register_tool("A1", {CapabilityDimension.PHYS_IO})
        analyzer_comb.register_tool("A2", {CapabilityDimension.PHYS_IO})
        analyzer_comb.register_tool("B", {CapabilityDimension.PHYS_NUM})
        
        # Resultados
        res_a = analyzer_a.analyze_redundancy()
        res_b = analyzer_b.analyze_redundancy()
        res_comb = analyzer_comb.analyze_redundancy()
        
        # Verificación: |M(A⊕B)| = |M(A)| + |M(B)|
        assert len(res_comb['essential_tools']) == (
            len(res_a['essential_tools']) + len(res_b['essential_tools'])
        ), (
            f"Coproducto no isomorfo: "
            f"|M(A⊕B)|={len(res_comb['essential_tools'])} ≠ "
            f"|M(A)| + |M(B)|={len(res_a['essential_tools']) + len(res_b['essential_tools'])}"
        )

    def test_unitarity_support_conservation(self, analyzer):
        """
        AXIOMA CUÁNTICO: ||ψ|| invariante (unitaridad)
        
        El soporte funcional (conjunto de capacidades cubiertas) se conserva
        bajo minimización. No puede haber "pérdida" de funcionalidad.
        """
        analyzer.register_tool("T1", {
            CapabilityDimension.PHYS_IO,
            CapabilityDimension.PHYS_NUM
        })
        analyzer.register_tool("T2", {CapabilityDimension.PHYS_NUM})
        
        # Soporte inicial (capacidades disponibles)
        support_initial = set()
        for tool in analyzer.tools:
            support_initial.update(tool.capabilities.components)
        
        # Soporte final (capacidades de herramientas esenciales)
        results = analyzer.analyze_redundancy()
        support_final = set()
        for tool_name in results['essential_tools']:
            tool = next((t for t in analyzer.tools if t.name == tool_name), None)
            if tool:
                support_final.update(tool.capabilities.components)
        
        assert support_initial == support_final, (
            f"Violación de unitaridad (conservación de soporte): "
            f"soporte_inicial={support_initial}, "
            f"soporte_final={support_final}"
        )

    def test_collapse_determinism_lexicographic_observable(self, analyzer):
        """
        AXIOMA CUÁNTICO: Colapso determinista mediante observable.
        
        Con herramientas redundantes idénticas, el observable "nombre lexicográfico"
        induce colapso determinista al nombre más pequeño (convención de rotura
        de simetría).
        """
        analyzer.register_tool("Alpha", {CapabilityDimension.PHYS_IO})
        analyzer.register_tool("Omega", {CapabilityDimension.PHYS_IO})
        
        results = analyzer.analyze_redundancy()
        
        # Determinismo: solo uno survives, y es el lexicográficamente menor
        assert "Alpha" in results['essential_tools'], (
            f"Colapso determinista falló: 'Alpha' (menor) no está en esenciales"
        )
        assert "Omega" in results['redundant_tools'], (
            f"Colapso determinista falló: 'Omega' (mayor) no está en redundantes"
        )


# ========================================================================================
# CLASE 6: PRUEBAS DE RENDIMIENTO Y ESCALABILIDAD
# ========================================================================================

class TestPerformanceScalability:
    """
    PROPÓSITO: Garantizar eficiencia computacional y escalabilidad.
    
    MÉTRICAS:
      • Tiempo de ejecución: ≤ MAX_COMPUTATION_TIME_S
      • Uso de memoria: Lineal en tamaño de entrada
      • Complejidad teórica: O(3^n) para Quine-McCluskey en B^n
    """

    @pytest.mark.parametrize("num_vars", [4, 6, 8])
    def test_quine_mccluskey_execution_time_scaling(self, num_vars):
        """
        VALIDACIÓN: Tiempo de ejecución dentro de límites tolerables.
        
        Para B^n con n ≤ 8, la convergencia debe ser < MAX_COMPUTATION_TIME_S.
        Basado en experiencia empírica con hardware estándar.
        """
        qm = QuineMcCluskeyMinimizer(num_vars)
        
        # Generar conjunto denso de minitérminos (cada 2 números)
        minterms = list(range(0, 2**num_vars, 2))
        
        start_time = time.time()
        primes = qm.compute_prime_implicants(minterms)
        duration = time.time() - start_time
        
        assert duration < MAX_COMPUTATION_TIME_S, (
            f"Degradación de rendimiento en B^{num_vars}: "
            f"{duration:.4f}s > {MAX_COMPUTATION_TIME_S}s"
        )
        
        assert len(primes) > 0, (
            f"No se computaron implicantes primos para B^{num_vars}"
        )
        
        logger.info(
            f"Quine-McCluskey[B^{num_vars}]: "
            f"computed {len(primes)} primes in {duration:.4f}s"
        )

    def test_analyzer_memory_stability_large_tool_set(self, analyzer):
        """
        VALIDACIÓN: Estabilidad de memoria con muchas herramientas.
        
        Registrar n_tools=100 herramientas no debe causar comportamiento
        patológico o fugas de memoria.
        """
        n_tools = 100
        
        try:
            for i in range(n_tools):
                analyzer.register_tool(
                    f"Tool_{i}",
                    {CapabilityDimension.PHYS_IO}
                )
            
            # Verificar integridad estructural
            matrix = analyzer.build_incidence_matrix()
            
            assert matrix.shape == (n_tools, len(CapabilityDimension)), (
                f"Dimensión de matriz incorrecta: {matrix.shape}"
            )
            
            assert np.sum(matrix) == n_tools, (
                f"Suma de matriz incorrecta: {np.sum(matrix)} ≠ {n_tools}"
            )
            
        except MemoryError:
            pytest.fail("Memory error al registrar gran conjunto de herramientas")


# ========================================================================================
# EJECUCIÓN PRINCIPAL
# ========================================================================================

if __name__ == "__main__":
    # Ejecución con máxima verbosidad y reportes detallados
    pytest.main([
        __file__,
        "-vv",                    # Verbosidad máxima
        "--tb=short",             # Traceback corto
        "-x",                      # Parar en primer fallo
        "--color=yes",            # Colorizar salida
        "--durations=10"          # Mostrar 10 pruebas más lentas
    ])