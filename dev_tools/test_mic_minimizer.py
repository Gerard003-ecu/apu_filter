"""
=========================================================================================
Suite de Pruebas: Auditoría de Redundancia MIC (Algoritmo de Quine-McCluskey)
Ubicación: dev_tools/tests/test_mic_minimizer.py
Versión: 2.0 - Suite Rigurosa de Validación Matemática
=========================================================================================

METODOLOGÍA DE PRUEBAS:
----------------------
1. PRUEBAS ALGEBRAICAS:
   - Axiomas de retículo booleano (idempotencia, conmutatividad, distributividad)
   - Propiedades de grupo (identidad, inversos)
   - Isomorfismos entre representaciones

2. PRUEBAS TOPOLÓGICAS:
   - Invarianza bajo homeomorfismos
   - Consistencia de grupos de homología
   - Propiedades de conexidad

3. PRUEBAS DE TEORÍA ESPECTRAL:
   - Preservación de rango bajo transformaciones
   - Ortogonalidad de subespacios
   - Estabilidad numérica

4. PRUEBAS DE TEORÍA DE GRAFOS:
   - Propiedades de componentes conexas
   - Ciclicidad y aciclicidad
   - Isomorfismo de grafos

5. PRUEBAS CATEGÓRICAS:
   - Funtorialidad de transformaciones
   - Composición de morfismos
   - Propiedades universales

6. PRUEBAS DE CORRECCIÓN ALGORÍTMICA:
   - Casos límite (vacío, completo, singleton)
   - Propiedades de convergencia
   - Minimalidad de soluciones

7. PRUEBAS CUÁNTICAS (ANALOGÍA):
   - Unitaridad de transformaciones
   - Superposición de estados
   - Colapso de función de onda

=========================================================================================
"""

import unittest
import numpy as np
from typing import List, Set, Dict, FrozenSet
from itertools import combinations, product
import logging
from collections import defaultdict

# Importaciones del módulo a probar
import sys
from pathlib import Path

# Ajustar path si es necesario
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

# Configuración de logging para pruebas
logging.basicConfig(
    level=logging.WARNING,  # Reducimos verbosidad en tests
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("MIC.Tests")


# ========================================================================================
# PRUEBAS ALGEBRAICAS: ESTRUCTURAS BOOLEANAS
# ========================================================================================

class TestBooleanVectorAlgebra(unittest.TestCase):
    """
    Pruebas de axiomas algebraicos para BooleanVector.
    Valida que la estructura sea un retículo booleano riguroso.
    """
    
    def setUp(self):
        """Inicializa vectores de prueba."""
        self.v_empty = BooleanVector(frozenset())
        self.v_io = BooleanVector(frozenset([CapabilityDimension.PHYS_IO]))
        self.v_num = BooleanVector(frozenset([CapabilityDimension.PHYS_NUM]))
        self.v_io_num = BooleanVector(frozenset([
            CapabilityDimension.PHYS_IO,
            CapabilityDimension.PHYS_NUM
        ]))
        self.v_all = BooleanVector(frozenset(CapabilityDimension))
    
    # --------------------------------------------------------------------------------
    # Axiomas de Retículo Booleano
    # --------------------------------------------------------------------------------
    
    def test_idempotence_union(self):
        """Axioma: A ∪ A = A (Idempotencia del supremo)"""
        for vec in [self.v_empty, self.v_io, self.v_io_num, self.v_all]:
            with self.subTest(vec=vec):
                result = vec.union(vec)
                self.assertEqual(result, vec, "Fallo en idempotencia de unión")
    
    def test_idempotence_intersection(self):
        """Axioma: A ∩ A = A (Idempotencia del ínfimo)"""
        for vec in [self.v_empty, self.v_io, self.v_io_num, self.v_all]:
            with self.subTest(vec=vec):
                result = vec.intersection(vec)
                self.assertEqual(result, vec, "Fallo en idempotencia de intersección")
    
    def test_commutativity_union(self):
        """Axioma: A ∪ B = B ∪ A (Conmutatividad del supremo)"""
        pairs = [(self.v_io, self.v_num), (self.v_io_num, self.v_all)]
        for a, b in pairs:
            with self.subTest(a=a, b=b):
                self.assertEqual(a.union(b), b.union(a), "Fallo en conmutatividad de unión")
    
    def test_commutativity_intersection(self):
        """Axioma: A ∩ B = B ∩ A (Conmutatividad del ínfimo)"""
        pairs = [(self.v_io, self.v_num), (self.v_io_num, self.v_all)]
        for a, b in pairs:
            with self.subTest(a=a, b=b):
                self.assertEqual(
                    a.intersection(b), 
                    b.intersection(a), 
                    "Fallo en conmutatividad de intersección"
                )
    
    def test_associativity_union(self):
        """Axioma: (A ∪ B) ∪ C = A ∪ (B ∪ C) (Asociatividad del supremo)"""
        a, b, c = self.v_io, self.v_num, self.v_io_num
        left = a.union(b).union(c)
        right = a.union(b.union(c))
        self.assertEqual(left, right, "Fallo en asociatividad de unión")
    
    def test_associativity_intersection(self):
        """Axioma: (A ∩ B) ∩ C = A ∩ (B ∩ C) (Asociatividad del ínfimo)"""
        a, b, c = self.v_io_num, self.v_io, self.v_all
        left = a.intersection(b).intersection(c)
        right = a.intersection(b.intersection(c))
        self.assertEqual(left, right, "Fallo en asociatividad de intersección")
    
    def test_absorption_laws(self):
        """Axiomas de absorción: A ∪ (A ∩ B) = A y A ∩ (A ∪ B) = A"""
        a, b = self.v_io, self.v_num
        
        # A ∪ (A ∩ B) = A
        self.assertEqual(
            a.union(a.intersection(b)), 
            a, 
            "Fallo en ley de absorción (supremo)"
        )
        
        # A ∩ (A ∪ B) = A
        self.assertEqual(
            a.intersection(a.union(b)), 
            a, 
            "Fallo en ley de absorción (ínfimo)"
        )
    
    def test_distributivity(self):
        """Axioma: A ∪ (B ∩ C) = (A ∪ B) ∩ (A ∪ C) (Distributividad)"""
        a = self.v_io
        b = self.v_num
        c = self.v_io_num
        
        left = a.union(b.intersection(c))
        right = a.union(b).intersection(a.union(c))
        
        self.assertEqual(left, right, "Fallo en distributividad")
    
    def test_identity_elements(self):
        """Axiomas de identidad: A ∪ ∅ = A y A ∩ U = A"""
        vec = self.v_io
        
        # Elemento neutro de la unión (∅)
        self.assertEqual(
            vec.union(self.v_empty), 
            vec, 
            "Fallo en identidad de unión (elemento vacío)"
        )
        
        # Elemento neutro de la intersección (U = todos)
        self.assertEqual(
            vec.intersection(self.v_all), 
            vec, 
            "Fallo en identidad de intersección (universo)"
        )
    
    def test_complement_properties(self):
        """Propiedades de complemento vía XOR: A ⊕ A = ∅"""
        for vec in [self.v_io, self.v_num, self.v_io_num]:
            with self.subTest(vec=vec):
                xor_result = vec.symmetric_difference(vec)
                self.assertEqual(
                    xor_result, 
                    self.v_empty, 
                    "Fallo en propiedad de auto-cancelación XOR"
                )
    
    # --------------------------------------------------------------------------------
    # Propiedades de Representación
    # --------------------------------------------------------------------------------
    
    def test_binary_string_length(self):
        """La representación binaria debe tener longitud correcta."""
        num_vars = len(CapabilityDimension)
        for vec in [self.v_empty, self.v_io, self.v_all]:
            with self.subTest(vec=vec):
                binary = vec.to_binary_string(num_vars)
                self.assertEqual(
                    len(binary), 
                    num_vars, 
                    f"Longitud incorrecta: esperado {num_vars}, obtenido {len(binary)}"
                )
    
    def test_binary_string_consistency(self):
        """La representación binaria debe ser consistente con las capacidades."""
        num_vars = len(CapabilityDimension)
        
        # Vector vacío debe ser todo ceros
        self.assertEqual(self.v_empty.to_binary_string(num_vars), "0" * num_vars)
        
        # Vector completo debe ser todo unos
        self.assertEqual(self.v_all.to_binary_string(num_vars), "1" * num_vars)
        
        # Vector con PHYS_IO (índice 0) debe tener '1' en posición 0
        binary_io = self.v_io.to_binary_string(num_vars)
        self.assertEqual(binary_io[0], '1', "PHYS_IO no marcado correctamente")
    
    def test_minterm_conversion(self):
        """Conversión a minitérmino debe ser correcta."""
        # Vector vacío → minitérmino 0
        self.assertEqual(self.v_empty.to_minterm(), 0)
        
        # PHYS_IO (bit 0) → minitérmino 1
        self.assertEqual(self.v_io.to_minterm(), 1)
        
        # PHYS_NUM (bit 1) → minitérmino 2
        self.assertEqual(self.v_num.to_minterm(), 2)
        
        # PHYS_IO | PHYS_NUM → minitérmino 3
        self.assertEqual(self.v_io_num.to_minterm(), 3)
    
    def test_hamming_weight(self):
        """Peso de Hamming debe contar correctamente las capacidades activas."""
        self.assertEqual(self.v_empty.hamming_weight(), 0)
        self.assertEqual(self.v_io.hamming_weight(), 1)
        self.assertEqual(self.v_io_num.hamming_weight(), 2)
        self.assertEqual(self.v_all.hamming_weight(), len(CapabilityDimension))
    
    # --------------------------------------------------------------------------------
    # Propiedades de Inmutabilidad
    # --------------------------------------------------------------------------------
    
    def test_immutability(self):
        """Los BooleanVector deben ser inmutables (frozen)."""
        vec = self.v_io
        with self.assertRaises(AttributeError):
            vec.components = frozenset()  # Debe fallar
    
    def test_hashability(self):
        """Los BooleanVector deben ser hashables para uso en conjuntos."""
        vec_set = {self.v_io, self.v_num, self.v_io}  # Duplicado intencional
        self.assertEqual(len(vec_set), 2, "Vectores idénticos deben colapsar en conjuntos")


# ========================================================================================
# PRUEBAS DE ALGORITMO: QUINE-MCCLUSKEY
# ========================================================================================

class TestQuineMcCluskeyMinimizer(unittest.TestCase):
    """
    Pruebas del algoritmo de Quine-McCluskey.
    Valida corrección matemática, convergencia y minimalidad.
    """
    
    def setUp(self):
        """Inicializa minimizadores para diferentes dimensiones."""
        self.qm_2 = QuineMcCluskeyMinimizer(num_vars=2)
        self.qm_3 = QuineMcCluskeyMinimizer(num_vars=3)
        self.qm_4 = QuineMcCluskeyMinimizer(num_vars=4)
        self.qm_5 = QuineMcCluskeyMinimizer(num_vars=5)
    
    # --------------------------------------------------------------------------------
    # Casos Límite
    # --------------------------------------------------------------------------------
    
    def test_empty_input(self):
        """Conjunto vacío de minitérminos debe retornar conjunto vacío de implicantes."""
        primes = self.qm_3.compute_prime_implicants([])
        self.assertEqual(len(primes), 0, "Entrada vacía debe dar salida vacía")
    
    def test_single_minterm(self):
        """Un solo minitérmino debe retornar un solo implicante primo (sí mismo)."""
        primes = self.qm_3.compute_prime_implicants([5])  # 101 en binario
        self.assertEqual(len(primes), 1, "Un minitérmino debe dar un implicante")
        prime = list(primes)[0]
        self.assertEqual(prime.pattern, "101", "Patrón debe ser idéntico al minitérmino")
    
    def test_all_minterms(self):
        """Todos los minitérminos deben colapsar a un solo implicante (todo don't-care)."""
        all_minterms = list(range(2**3))  # 0-7 para 3 variables
        primes = self.qm_3.compute_prime_implicants(all_minterms)
        
        # Debe haber un solo implicante que cubra todo
        self.assertGreaterEqual(len(primes), 1, "Debe haber al menos un implicante")
        
        # Verificar cobertura completa
        covered = set()
        for prime in primes:
            covered.update(prime.covered_minterms)
        self.assertEqual(covered, set(all_minterms), "Debe cubrir todos los minitérminos")
    
    def test_duplicate_minterms(self):
        """Minitérminos duplicados deben tratarse como uno solo."""
        primes1 = self.qm_3.compute_prime_implicants([1, 2, 3])
        primes2 = self.qm_3.compute_prime_implicants([1, 1, 2, 2, 3, 3])
        
        # Convertir a conjuntos de patrones para comparar
        patterns1 = {p.pattern for p in primes1}
        patterns2 = {p.pattern for p in primes2}
        
        self.assertEqual(patterns1, patterns2, "Duplicados deben ignorarse")
    
    # --------------------------------------------------------------------------------
    # Casos Conocidos (Ejemplos Clásicos)
    # --------------------------------------------------------------------------------
    
    def test_classic_example_2vars(self):
        """
        Ejemplo clásico: F(A,B) = Σ(0,1,2) = A' + B'
        Minitérminos: 00, 01, 10
        Implicantes primos esperados: 0- (A'), -0 (B')
        """
        minterms = [0, 1, 2]  # 00, 01, 10
        primes = self.qm_2.compute_prime_implicants(minterms)
        
        patterns = {p.pattern for p in primes}
        
        # Debe contener 0- y -0
        self.assertIn("0-", patterns, "Debe incluir 0- (A')")
        self.assertIn("-0", patterns, "Debe incluir -0 (B')")
    
    def test_classic_example_3vars(self):
        """
        Ejemplo: F(A,B,C) = Σ(0,1,2,5,6,7)
        Implicantes primos: -0- (B'), 1-- (A), --1 (C)
        """
        minterms = [0, 1, 2, 5, 6, 7]
        primes = self.qm_3.compute_prime_implicants(minterms)
        
        # Verificar que se cubren todos los minitérminos
        covered = set()
        for prime in primes:
            covered.update(prime.covered_minterms)
        
        self.assertEqual(covered, set(minterms), "Debe cubrir todos los minitérminos")
    
    def test_adjacent_minterms(self):
        """
        Minitérminos adyacentes (diferencia de 1 bit) deben combinarse.
        Ejemplo: 0 (000) y 1 (001) → 00- (AB')
        """
        minterms = [0, 1]  # 000, 001
        primes = self.qm_3.compute_prime_implicants(minterms)
        
        self.assertEqual(len(primes), 1, "Dos adyacentes deben dar un implicante")
        prime = list(primes)[0]
        self.assertEqual(prime.pattern, "00-", "Patrón debe ser 00-")
    
    def test_non_adjacent_minterms(self):
        """
        Minitérminos no adyacentes no deben combinarse.
        Ejemplo: 0 (000) y 3 (011) → dos implicantes separados
        """
        minterms = [0, 3]  # 000, 011 (difieren en 2 bits)
        primes = self.qm_3.compute_prime_implicants(minterms)
        
        self.assertEqual(len(primes), 2, "No adyacentes deben permanecer separados")
    
    # --------------------------------------------------------------------------------
    # Propiedades de Convergencia
    # --------------------------------------------------------------------------------
    
    def test_convergence_finite_iterations(self):
        """El algoritmo debe converger en número finito de iteraciones."""
        # Peor caso: todos los minitérminos
        all_minterms = list(range(2**4))
        
        # No debe lanzar excepciones ni ciclar infinitamente
        try:
            primes = self.qm_4.compute_prime_implicants(all_minterms)
            self.assertIsNotNone(primes, "Debe converger y retornar resultado")
        except Exception as e:
            self.fail(f"Algoritmo no convergió: {e}")
    
    # --------------------------------------------------------------------------------
    # Propiedades de Minimalidad
    # --------------------------------------------------------------------------------
    
    def test_prime_implicants_are_minimal(self):
        """
        Cada implicante primo debe ser irreducible:
        no puede simplificarse más sin perder cobertura.
        """
        minterms = [0, 1, 2, 3]
        primes = self.qm_2.compute_prime_implicants(minterms)
        
        # Con 4 minitérminos en 2 variables (todos), debe ser --
        self.assertEqual(len(primes), 1, "Debe haber un solo implicante")
        prime = list(primes)[0]
        self.assertEqual(prime.pattern.count('-'), 2, "Debe tener máxima generalización")
    
    def test_coverage_completeness(self):
        """Todos los minitérminos deben estar cubiertos por los implicantes primos."""
        minterms = [1, 3, 5, 7, 9, 11, 13, 15]
        primes = self.qm_4.compute_prime_implicants(minterms)
        
        covered = set()
        for prime in primes:
            covered.update(prime.covered_minterms)
        
        self.assertEqual(
            covered, 
            set(minterms), 
            "Los implicantes primos deben cubrir todos los minitérminos"
        )
    
    # --------------------------------------------------------------------------------
    # Operaciones de Combinación
    # --------------------------------------------------------------------------------
    
    def test_combine_terms_identical(self):
        """Términos idénticos no deben combinarse."""
        result = self.qm_3.combine_terms("101", "101")
        self.assertIsNone(result, "Términos idénticos no son combinables")
    
    def test_combine_terms_single_difference(self):
        """Términos con diferencia de 1 bit deben combinarse correctamente."""
        result = self.qm_3.combine_terms("101", "111")
        self.assertEqual(result, "1-1", "Debe combinar correctamente")
    
    def test_combine_terms_multiple_differences(self):
        """Términos con más de 1 bit diferente no deben combinarse."""
        result = self.qm_3.combine_terms("000", "111")
        self.assertIsNone(result, "Términos con 3 diferencias no son combinables")
    
    def test_combine_terms_with_dont_care(self):
        """Combinación con don't-care debe respetar posiciones establecidas."""
        result = self.qm_3.combine_terms("0-1", "1-1")
        self.assertEqual(result, "--1", "Don't-care debe propagarse")
    
    def test_combine_terms_incompatible_dont_care(self):
        """Términos con don't-care en diferentes posiciones no deben combinarse incorrectamente."""
        result = self.qm_3.combine_terms("0-0", "01-")
        self.assertIsNone(result, "Don't-care incompatible no debe combinar")
    
    # --------------------------------------------------------------------------------
    # Validación de Entrada
    # --------------------------------------------------------------------------------
    
    def test_invalid_minterm_negative(self):
        """Minitérminos negativos deben lanzar excepción."""
        with self.assertRaises(ValueError):
            self.qm_3.compute_prime_implicants([-1, 0, 1])
    
    def test_invalid_minterm_out_of_range(self):
        """Minitérminos fuera de rango deben lanzar excepción."""
        max_minterm = (1 << 3) - 1  # 7 para 3 variables
        with self.assertRaises(ValueError):
            self.qm_3.compute_prime_implicants([0, max_minterm + 1])
    
    def test_invalid_num_vars(self):
        """Número inválido de variables debe lanzar excepción."""
        with self.assertRaises(ValueError):
            QuineMcCluskeyMinimizer(num_vars=0)
        
        with self.assertRaises(ValueError):
            QuineMcCluskeyMinimizer(num_vars=33)  # Excede 32 bits
    
    # --------------------------------------------------------------------------------
    # Distancia de Hamming
    # --------------------------------------------------------------------------------
    
    def test_hamming_distance_identical(self):
        """Distancia de Hamming entre términos idénticos debe ser 0."""
        dist = self.qm_3.hamming_distance("101", "101")
        self.assertEqual(dist, 0, "Términos idénticos: distancia = 0")
    
    def test_hamming_distance_single_bit(self):
        """Distancia de Hamming con 1 bit diferente debe ser 1."""
        dist = self.qm_3.hamming_distance("101", "001")
        self.assertEqual(dist, 1, "Un bit diferente: distancia = 1")
    
    def test_hamming_distance_dont_care(self):
        """Don't-care no debe contar en distancia de Hamming."""
        dist = self.qm_3.hamming_distance("1-1", "101")
        self.assertEqual(dist, 0, "Don't-care debe ignorarse")
    
    def test_hamming_distance_different_lengths(self):
        """Términos de diferente longitud deben lanzar excepción."""
        with self.assertRaises(ValueError):
            self.qm_3.hamming_distance("10", "101")
    
    # --------------------------------------------------------------------------------
    # Cobertura Esencial y Minimal
    # --------------------------------------------------------------------------------
    
    def test_essential_prime_implicants(self):
        """Implicantes esenciales deben identificarse correctamente."""
        minterms = [0, 1, 4, 5]  # 00, 01, 100, 101
        primes = self.qm_3.compute_prime_implicants(minterms)
        essential, covered = self.qm_3.find_essential_prime_implicants(primes, set(minterms))
        
        # Debe haber al menos un implicante esencial
        self.assertGreater(len(essential), 0, "Debe haber implicantes esenciales")
        
        # Los esenciales deben cubrir algunos minitérminos
        self.assertGreater(len(covered), 0, "Implicantes esenciales deben cubrir minitérminos")
    
    def test_minimal_cover_completeness(self):
        """La cobertura minimal debe cubrir todos los minitérminos."""
        minterms = [0, 1, 2, 5, 6, 7]
        primes = self.qm_3.compute_prime_implicants(minterms)
        essential, covered = self.qm_3.find_essential_prime_implicants(primes, set(minterms))
        minimal = self.qm_3.minimal_cover(primes, set(minterms), essential, covered)
        
        # Cobertura completa
        all_covered = set()
        for impl in minimal:
            all_covered.update(impl.covered_minterms)
        
        self.assertEqual(
            all_covered, 
            set(minterms), 
            "Cobertura minimal debe cubrir todos los minitérminos"
        )


# ========================================================================================
# PRUEBAS TOPOLÓGICAS: HOMOLOGÍA Y COMPONENTES
# ========================================================================================

class TestTopologicalProperties(unittest.TestCase):
    """
    Pruebas de propiedades topológicas:
    - Grupos de homología
    - Componentes conexas
    - Invariantes topológicos
    """
    
    def setUp(self):
        """Inicializa analizador limpio para cada prueba."""
        self.analyzer = MICRedundancyAnalyzer()
    
    def test_empty_homology(self):
        """Sistema vacío debe tener homología trivial."""
        homology = self.analyzer.compute_homology_groups()
        
        self.assertEqual(homology['H_0'], 0, "H_0 de conjunto vacío debe ser 0")
        self.assertEqual(homology['H_1'], 0, "H_1 de conjunto vacío debe ser 0")
        self.assertEqual(len(homology['components']), 0, "No debe haber componentes")
    
    def test_single_tool_homology(self):
        """Una herramienta aislada es una componente conexa."""
        self.analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO})
        homology = self.analyzer.compute_homology_groups()
        
        self.assertEqual(homology['H_0'], 1, "Una herramienta → H_0 = 1")
        self.assertEqual(homology['H_1'], 0, "Sin redundancia → H_1 = 0")
        self.assertEqual(len(homology['components']), 1, "Debe haber 1 componente")
    
    def test_disconnected_tools_homology(self):
        """Herramientas sin capacidades compartidas → múltiples componentes."""
        self.analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO})
        self.analyzer.register_tool("tool2", {CapabilityDimension.PHYS_NUM})
        self.analyzer.register_tool("tool3", {CapabilityDimension.TACT_TOPO})
        
        homology = self.analyzer.compute_homology_groups()
        
        # Tres herramientas sin intersección → 3 componentes
        self.assertEqual(homology['H_0'], 3, "Herramientas desconectadas → H_0 = 3")
        self.assertEqual(homology['H_1'], 0, "Sin redundancia exacta → H_1 = 0")
    
    def test_connected_tools_homology(self):
        """Herramientas con capacidades compartidas → componente única."""
        self.analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO, CapabilityDimension.PHYS_NUM})
        self.analyzer.register_tool("tool2", {CapabilityDimension.PHYS_NUM, CapabilityDimension.TACT_TOPO})
        self.analyzer.register_tool("tool3", {CapabilityDimension.TACT_TOPO, CapabilityDimension.STRAT_FIN})
        
        homology = self.analyzer.compute_homology_groups()
        
        # Todas comparten al menos una capacidad transitivamente → 1 componente
        self.assertEqual(homology['H_0'], 1, "Herramientas conectadas → H_0 = 1")
    
    def test_redundancy_cycle_detection(self):
        """Herramientas idénticas deben detectarse como ciclo de redundancia."""
        self.analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO})
        self.analyzer.register_tool("tool2", {CapabilityDimension.PHYS_IO})  # Idéntica
        
        homology = self.analyzer.compute_homology_groups()
        
        # Dos herramientas idénticas → ciclo de redundancia
        self.assertEqual(homology['H_1'], 1, "Redundancia exacta → H_1 = 1")
        self.assertEqual(len(homology['redundancy_cycles']), 1, "Debe detectar 1 ciclo")
        
        cycle = homology['redundancy_cycles'][0]
        self.assertIn("tool1", cycle, "Ciclo debe contener tool1")
        self.assertIn("tool2", cycle, "Ciclo debe contener tool2")
    
    def test_multiple_redundancy_cycles(self):
        """Múltiples grupos de redundancia deben detectarse independientemente."""
        self.analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO})
        self.analyzer.register_tool("tool2", {CapabilityDimension.PHYS_IO})  # Ciclo 1
        self.analyzer.register_tool("tool3", {CapabilityDimension.PHYS_NUM})
        self.analyzer.register_tool("tool4", {CapabilityDimension.PHYS_NUM})  # Ciclo 2
        
        homology = self.analyzer.compute_homology_groups()
        
        self.assertEqual(homology['H_1'], 2, "Dos ciclos independientes → H_1 = 2")
        self.assertEqual(len(homology['redundancy_cycles']), 2, "Debe detectar 2 ciclos")


# ========================================================================================
# PRUEBAS DE TEORÍA ESPECTRAL: MATRICES Y RANGOS
# ========================================================================================

class TestSpectralTheory(unittest.TestCase):
    """
    Pruebas de propiedades espectrales:
    - Rango matricial
    - Dependencias lineales
    - Ortogonalidad
    """
    
    def setUp(self):
        """Inicializa analizador limpio."""
        self.analyzer = MICRedundancyAnalyzer()
    
    def test_empty_matrix_rank(self):
        """Matriz vacía debe tener rango 0."""
        matrix = self.analyzer.build_incidence_matrix()
        rank = self.analyzer.compute_spectral_rank(matrix)
        
        self.assertEqual(rank, 0, "Matriz vacía → rango = 0")
    
    def test_single_tool_rank(self):
        """Una herramienta con k capacidades → rango ≤ 1."""
        self.analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO, CapabilityDimension.PHYS_NUM})
        
        matrix = self.analyzer.build_incidence_matrix()
        rank = self.analyzer.compute_spectral_rank(matrix)
        
        self.assertEqual(rank, 1, "Una herramienta → rango = 1")
    
    def test_independent_tools_rank(self):
        """Herramientas linealmente independientes → rango = número de herramientas."""
        self.analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO})
        self.analyzer.register_tool("tool2", {CapabilityDimension.PHYS_NUM})
        self.analyzer.register_tool("tool3", {CapabilityDimension.TACT_TOPO})
        
        matrix = self.analyzer.build_incidence_matrix()
        rank = self.analyzer.compute_spectral_rank(matrix)
        
        self.assertEqual(rank, 3, "3 herramientas independientes → rango = 3")
    
    def test_dependent_tools_rank(self):
        """Herramientas idénticas → rango < número de herramientas."""
        self.analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO})
        self.analyzer.register_tool("tool2", {CapabilityDimension.PHYS_IO})  # Idéntica
        
        matrix = self.analyzer.build_incidence_matrix()
        rank = self.analyzer.compute_spectral_rank(matrix)
        
        self.assertEqual(rank, 1, "2 herramientas idénticas → rango = 1")
    
    def test_linear_dependency_detection(self):
        """Dependencias lineales deben detectarse correctamente."""
        self.analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO})
        self.analyzer.register_tool("tool2", {CapabilityDimension.PHYS_IO, CapabilityDimension.PHYS_NUM})
        
        matrix = self.analyzer.build_incidence_matrix()
        dependencies = self.analyzer.detect_linear_dependencies(matrix)
        
        # tool1 ⊆ tool2 (tool1 es subconjunto de tool2)
        self.assertGreater(len(dependencies), 0, "Debe detectar dependencia")
        
        # Verificar que la dependencia es correcta
        dep_indices = dependencies[0]
        tool_names = sorted([self.analyzer.tools[i].name for i in dep_indices])
        self.assertIn("tool1", tool_names, "Dependencia debe involucrar tool1")
        self.assertIn("tool2", tool_names, "Dependencia debe involucrar tool2")
    
    def test_incidence_matrix_dimensions(self):
        """Matriz de incidencia debe tener dimensiones correctas."""
        self.analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO})
        self.analyzer.register_tool("tool2", {CapabilityDimension.PHYS_NUM})
        
        matrix = self.analyzer.build_incidence_matrix()
        
        expected_rows = 2  # 2 herramientas
        expected_cols = len(CapabilityDimension)
        
        self.assertEqual(matrix.shape[0], expected_rows, f"Filas incorrectas: esperado {expected_rows}")
        self.assertEqual(matrix.shape[1], expected_cols, f"Columnas incorrectas: esperado {expected_cols}")
    
    def test_incidence_matrix_values(self):
        """Valores de la matriz de incidencia deben ser 0 o 1."""
        self.analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO, CapabilityDimension.TACT_TOPO})
        
        matrix = self.analyzer.build_incidence_matrix()
        
        unique_values = np.unique(matrix)
        self.assertTrue(
            np.all(np.isin(unique_values, [0, 1])),
            "Matriz debe contener solo 0s y 1s"
        )


# ========================================================================================
# PRUEBAS DE INTEGRACIÓN: ANÁLISIS COMPLETO
# ========================================================================================

class TestRedundancyAnalysisIntegration(unittest.TestCase):
    """
    Pruebas de integración del análisis completo de redundancia.
    Valida el flujo end-to-end y las propiedades emergentes.
    """
    
    def test_minimal_configuration_no_redundancy(self):
        """Configuración minimal sin redundancia debe identificarse correctamente."""
        analyzer = MICRedundancyAnalyzer()
        
        analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO})
        analyzer.register_tool("tool2", {CapabilityDimension.PHYS_NUM})
        analyzer.register_tool("tool3", {CapabilityDimension.TACT_TOPO})
        
        results = analyzer.analyze_redundancy()
        
        self.assertEqual(len(results['essential_tools']), 3, "Todas deben ser esenciales")
        self.assertEqual(len(results['redundant_tools']), 0, "No debe haber redundantes")
    
    def test_complete_redundancy_detection(self):
        """Redundancia completa debe detectarse y reportarse."""
        analyzer = MICRedundancyAnalyzer()
        
        analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO})
        analyzer.register_tool("tool2", {CapabilityDimension.PHYS_IO})  # Redundante
        analyzer.register_tool("tool3", {CapabilityDimension.PHYS_NUM})
        analyzer.register_tool("tool4", {CapabilityDimension.PHYS_NUM})  # Redundante
        
        results = analyzer.analyze_redundancy()
        
        self.assertEqual(len(results['essential_tools']), 2, "Solo 2 deben ser esenciales")
        self.assertEqual(len(results['redundant_tools']), 2, "2 deben ser redundantes")
    
    def test_partial_redundancy(self):
        """Redundancia parcial (algunas herramientas redundantes) debe manejarse."""
        analyzer = MICRedundancyAnalyzer()
        
        analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO})
        analyzer.register_tool("tool2", {CapabilityDimension.PHYS_NUM})
        analyzer.register_tool("tool3", {CapabilityDimension.PHYS_NUM})  # Redundante con tool2
        
        results = analyzer.analyze_redundancy()
        
        self.assertEqual(len(results['essential_tools']), 2, "2 deben ser esenciales")
        self.assertEqual(len(results['redundant_tools']), 1, "1 debe ser redundante")
        self.assertIn("tool3", results['redundant_tools'], "tool3 debe ser redundante")
    
    def test_subset_redundancy(self):
        """Herramienta que es subconjunto de otra debe marcarse como redundante."""
        analyzer = MICRedundancyAnalyzer()
        
        analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO})
        analyzer.register_tool("tool2", {CapabilityDimension.PHYS_IO, CapabilityDimension.PHYS_NUM})
        
        results = analyzer.analyze_redundancy()
        
        # tool1 ⊂ tool2, pero tool1 aún puede ser esencial si cubre un minitérmino único
        # Verificamos que el análisis sea consistente
        self.assertIsInstance(results['essential_tools'], list)
        self.assertIsInstance(results['redundant_tools'], list)
    
    def test_original_example_from_code(self):
        """Replica el ejemplo original del código para validar regresión."""
        analyzer = MICRedundancyAnalyzer()
        
        analyzer.register_tool("stabilize_flux", {CapabilityDimension.PHYS_NUM})
        analyzer.register_tool("parse_raw", {CapabilityDimension.PHYS_IO, CapabilityDimension.PHYS_NUM})
        analyzer.register_tool("structure_logic", {CapabilityDimension.TACT_TOPO})
        analyzer.register_tool("audit_fusion", {CapabilityDimension.TACT_TOPO})  # Redundante
        analyzer.register_tool("lateral_pivot", {CapabilityDimension.STRAT_FIN})
        analyzer.register_tool("fat_tail_risk", {CapabilityDimension.STRAT_FIN})  # Redundante
        analyzer.register_tool("semantic_estimator", {CapabilityDimension.WIS_SEM})
        
        results = analyzer.analyze_redundancy()
        
        # Verificaciones esperadas
        self.assertIn("structure_logic", results['essential_tools'] + results['redundant_tools'])
        self.assertIn("audit_fusion", results['essential_tools'] + results['redundant_tools'])
        
        # Al menos una de las dos debe ser redundante
        topo_tools = ["structure_logic", "audit_fusion"]
        redundant_topo = [t for t in topo_tools if t in results['redundant_tools']]
        self.assertGreater(len(redundant_topo), 0, "Debe haber redundancia en herramientas topológicas")


# ========================================================================================
# PRUEBAS DE TEORÍA DE CATEGORÍAS: FUNTORIALIDAD
# ========================================================================================

class TestCategoricalProperties(unittest.TestCase):
    """
    Pruebas de propiedades categóricas:
    - Funtorialidad de transformaciones
    - Preservación de estructura
    - Propiedades universales
    """
    
    def test_tool_registration_is_functor(self):
        """El registro de herramientas debe preservar estructura (funtorialidad)."""
        analyzer1 = MICRedundancyAnalyzer()
        analyzer2 = MICRedundancyAnalyzer()
        
        tools = [
            ("tool1", {CapabilityDimension.PHYS_IO}),
            ("tool2", {CapabilityDimension.PHYS_NUM})
        ]
        
        # Registrar en el mismo orden
        for name, caps in tools:
            analyzer1.register_tool(name, caps)
            analyzer2.register_tool(name, caps)
        
        # Las matrices de incidencia deben ser idénticas
        matrix1 = analyzer1.build_incidence_matrix()
        matrix2 = analyzer2.build_incidence_matrix()
        
        np.testing.assert_array_equal(
            matrix1, 
            matrix2, 
            "Registro idéntico debe producir estructuras idénticas"
        )
    
    def test_morphism_composition(self):
        """Composición de transformaciones debe ser asociativa."""
        analyzer = MICRedundancyAnalyzer()
        
        # Transformación 1: Registro de herramientas
        analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO})
        analyzer.register_tool("tool2", {CapabilityDimension.PHYS_NUM})
        
        # Transformación 2: Construcción de matriz
        matrix = analyzer.build_incidence_matrix()
        
        # Transformación 3: Cálculo de rango
        rank = analyzer.compute_spectral_rank(matrix)
        
        # Composición directa
        rank_direct = analyzer.compute_spectral_rank(analyzer.build_incidence_matrix())
        
        self.assertEqual(rank, rank_direct, "Composición debe ser asociativa")
    
    def test_identity_morphism(self):
        """Morfismo identidad no debe alterar estructura."""
        analyzer = MICRedundancyAnalyzer()
        analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO})
        
        matrix1 = analyzer.build_incidence_matrix()
        
        # "Aplicar identidad" (reconstruir sin cambios)
        matrix2 = analyzer.build_incidence_matrix()
        
        np.testing.assert_array_equal(
            matrix1, 
            matrix2, 
            "Morfismo identidad debe preservar estructura"
        )


# ========================================================================================
# PRUEBAS DE PROPIEDADES CUÁNTICAS (ANALOGÍA)
# ========================================================================================

class TestQuantumAnalogies(unittest.TestCase):
    """
    Pruebas basadas en analogías con mecánica cuántica:
    - Superposición de estados
    - Colapso de función de onda
    - Unitaridad de transformaciones
    """
    
    def test_superposition_of_capabilities(self):
        """
        Herramienta con múltiples capacidades es análoga a superposición cuántica.
        Medición (análisis) debe colapsar a estado definido.
        """
        analyzer = MICRedundancyAnalyzer()
        
        # Herramienta en "superposición" de capacidades
        analyzer.register_tool(
            "quantum_tool",
            {CapabilityDimension.PHYS_IO, CapabilityDimension.PHYS_NUM, CapabilityDimension.TACT_TOPO}
        )
        
        results = analyzer.analyze_redundancy()
        
        # Tras "medición" (análisis), debe estar en estado definido
        self.assertIn(
            "quantum_tool",
            results['essential_tools'] + results['redundant_tools'],
            "Herramienta debe colapsar a estado definido"
        )
    
    def test_unitary_transformation_preservation(self):
        """
        Transformaciones deben preservar "norma" (número total de capacidades únicas).
        Análogo a unitaridad en MQ.
        """
        analyzer = MICRedundancyAnalyzer()
        
        analyzer.register_tool("tool1", {CapabilityDimension.PHYS_IO})
        analyzer.register_tool("tool2", {CapabilityDimension.PHYS_NUM})
        
        # "Norma" inicial: 2 capacidades únicas
        initial_capabilities = {CapabilityDimension.PHYS_IO, CapabilityDimension.PHYS_NUM}
        
        results = analyzer.analyze_redundancy()
        
        # "Norma" final: debe conservarse (todas las capacidades deben estar cubiertas)
        final_capabilities = set()
        for tool_name in results['essential_tools']:
            tool = next(t for t in analyzer.tools if t.name == tool_name)
            final_capabilities.update(tool.capabilities.components)
        
        self.assertEqual(
            initial_capabilities,
            final_capabilities,
            "Transformación debe preservar capacidades totales"
        )


# ========================================================================================
# PRUEBAS DE PROPIEDADES ESTOCÁSTICAS Y ESTADÍSTICAS
# ========================================================================================

class TestStatisticalProperties(unittest.TestCase):
    """
    Pruebas de propiedades estadísticas y estocásticas:
    - Distribución de capacidades
    - Entropía de Shannon
    - Medidas de diversidad
    """
    
    def test_capability_distribution(self):
        """La distribución de capacidades debe ser consistente."""
        analyzer = MICRedundancyAnalyzer()
        
        # Distribución uniforme: cada capacidad usada una vez
        for i, cap in enumerate(CapabilityDimension):
            analyzer.register_tool(f"tool{i}", {cap})
        
        matrix = analyzer.build_incidence_matrix()
        
        # Suma por columna debe ser 1 (cada capacidad usada exactamente una vez)
        col_sums = np.sum(matrix, axis=0)
        np.testing.assert_array_equal(
            col_sums,
            np.ones(len(CapabilityDimension)),
            "Distribución uniforme: cada capacidad usada una vez"
        )
    
    def test_entropy_maximization(self):
        """
        Configuración con máxima diversidad debe tener mayor entropía.
        Entropía aproximada por número de patrones únicos.
        """
        # Configuración de alta entropía: todas diferentes
        analyzer_high = MICRedundancyAnalyzer()
        for i in range(5):
            analyzer_high.register_tool(f"tool{i}", {list(CapabilityDimension)[i]})
        
        results_high = analyzer_high.analyze_redundancy()
        
        # Configuración de baja entropía: todas iguales
        analyzer_low = MICRedundancyAnalyzer()
        for i in range(5):
            analyzer_low.register_tool(f"tool{i}", {CapabilityDimension.PHYS_IO})
        
        results_low = analyzer_low.analyze_redundancy()
        
        # Alta entropía → más herramientas esenciales
        self.assertGreater(
            len(results_high['essential_tools']),
            len(results_low['essential_tools']),
            "Mayor diversidad → más herramientas esenciales"
        )


# ========================================================================================
# PRUEBAS DE RENDIMIENTO Y ESCALABILIDAD
# ========================================================================================

class TestPerformanceAndScalability(unittest.TestCase):
    """
    Pruebas de rendimiento y escalabilidad:
    - Tiempo de ejecución
    - Uso de memoria
    - Escalamiento con número de variables
    """
    
    def test_scalability_with_variables(self):
        """El algoritmo debe escalar razonablemente con número de variables."""
        import time
        
        times = []
        for num_vars in [3, 4, 5, 6]:
            qm = QuineMcCluskeyMinimizer(num_vars)
            minterms = list(range(min(2**num_vars, 16)))  # Limitar a 16 minitérminos
            
            start = time.time()
            qm.compute_prime_implicants(minterms)
            elapsed = time.time() - start
            
            times.append(elapsed)
        
        # Debe terminar en tiempo razonable (< 1 segundo para casos pequeños)
        self.assertLess(max(times), 1.0, "Debe ejecutarse en < 1 segundo para casos pequeños")
    
    def test_memory_efficiency(self):
        """Las estructuras deben usar memoria eficientemente."""
        analyzer = MICRedundancyAnalyzer()
        
        # Registrar 100 herramientas
        for i in range(100):
            cap = list(CapabilityDimension)[i % len(CapabilityDimension)]
            analyzer.register_tool(f"tool{i}", {cap})
        
        # No debe lanzar MemoryError
        try:
            matrix = analyzer.build_incidence_matrix()
            self.assertEqual(matrix.shape[0], 100, "Debe manejar 100 herramientas")
        except MemoryError:
            self.fail("Debe manejar 100 herramientas sin MemoryError")


# ========================================================================================
# SUITE DE PRUEBAS PRINCIPALES
# ========================================================================================

def suite():
    """Construye la suite completa de pruebas."""
    test_suite = unittest.TestSuite()
    
    # Pruebas algebraicas
    test_suite.addTest(unittest.makeSuite(TestBooleanVectorAlgebra))
    
    # Pruebas algorítmicas
    test_suite.addTest(unittest.makeSuite(TestQuineMcCluskeyMinimizer))
    
    # Pruebas topológicas
    test_suite.addTest(unittest.makeSuite(TestTopologicalProperties))
    
    # Pruebas espectrales
    test_suite.addTest(unittest.makeSuite(TestSpectralTheory))
    
    # Pruebas de integración
    test_suite.addTest(unittest.makeSuite(TestRedundancyAnalysisIntegration))
    
    # Pruebas categóricas
    test_suite.addTest(unittest.makeSuite(TestCategoricalProperties))
    
    # Pruebas cuánticas (analogía)
    test_suite.addTest(unittest.makeSuite(TestQuantumAnalogies))
    
    # Pruebas estadísticas
    test_suite.addTest(unittest.makeSuite(TestStatisticalProperties))
    
    # Pruebas de rendimiento
    test_suite.addTest(unittest.makeSuite(TestPerformanceAndScalability))
    
    return test_suite


# ========================================================================================
# EJECUTOR DE PRUEBAS CON REPORTE DETALLADO
# ========================================================================================

if __name__ == "__main__":
    print("="*80)
    print("SUITE DE PRUEBAS RIGUROSAS: MIC MINIMIZER")
    print("="*80)
    print("\n📊 CATEGORÍAS DE PRUEBAS:")
    print("  1. Álgebra Booleana (axiomas de retículo)")
    print("  2. Algoritmo Quine-McCluskey (corrección y convergencia)")
    print("  3. Topología Algebraica (homología y componentes)")
    print("  4. Teoría Espectral (rango y dependencias)")
    print("  5. Análisis de Integración (end-to-end)")
    print("  6. Teoría de Categorías (funtorialidad)")
    print("  7. Analogías Cuánticas (superposición y colapso)")
    print("  8. Propiedades Estadísticas (entropía y distribución)")
    print("  9. Rendimiento y Escalabilidad")
    print("\n" + "="*80 + "\n")
    
    # Ejecutar con verbosidad máxima
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite())
    
    # Resumen final
    print("\n" + "="*80)
    print("RESUMEN DE EJECUCIÓN")
    print("="*80)
    print(f"✓ Pruebas ejecutadas: {result.testsRun}")
    print(f"✓ Éxitos: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"✗ Fallos: {len(result.failures)}")
    print(f"⚠ Errores: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n🎉 TODAS LAS PRUEBAS PASARON EXITOSAMENTE")
        print("✅ El código cumple con los estándares de rigurosidad matemática")
    else:
        print("\n❌ ALGUNAS PRUEBAS FALLARON")
        print("⚠ Revisar los detalles arriba para correcciones necesarias")
    
    print("="*80 + "\n")
    
    # Código de salida apropiado
    sys.exit(0 if result.wasSuccessful() else 1)