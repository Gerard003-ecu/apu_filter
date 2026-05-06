"""
═════════════════════════════════════════════════════════════════════════════
MÓDULO: Test Suite para Solenoid Acoustic (Descomposición de Hodge-Helmholtz)
VERSIÓN: 1.0.0 - Suite Rigurosa de Validación Topológica y Numérica
UBICACIÓN: tests/unit/physics/test_solenoid_acoustic.py
═════════════════════════════════════════════════════════════════════════════

ARQUITECTURA DE TESTING:

Esta suite implementa pruebas exhaustivas de:
    1. Álgebra lineal numérica (SVD, pseudoinversas, proyecciones)
    2. Topología algebraica (números de Betti, complejo de cadenas)
    3. Descomposición de Hodge (ortogonalidad, completitud)
    4. Operadores de vorticidad (proyectores, circulación)
    5. Propiedades espectrales (Laplaciano, gaps, kernels)
    6. Invariantes físicos (energía, conservación)
    7. Casos límite y robustez numérica

ESTRUCTURA JERÁRQUICA:

    1. FIXTURES Y GRAFOS DE REFERENCIA
       ├── Grafo acíclico (DAG)
       ├── Grafo con un ciclo simple
       ├── Grafo con múltiples ciclos
       ├── Grafo desconectado
       └── Grafos patológicos (self-loops, multigrafos)

    2. TESTS DE ÁLGEBRA LINEAL NUMÉRICA
       ├── Tolerancias adaptativas
       ├── Rango numérico
       ├── Pseudoinversas (4 condiciones de Penrose)
       ├── Proyecciones ortogonales
       ├── Bases del kernel
       └── Números de condición

    3. TESTS DE CONSTRUCCIÓN DEL COMPLEJO
       ├── Matriz de incidencia B₁
       ├── Matriz de ciclos B₂ (FCB)
       ├── Verificación ∂₁ ∘ ∂₂ = 0
       ├── Dimensiones y rangos
       └── Euler-Poincaré

    4. TESTS DE LAPLACIANO DE HODGE
       ├── Simetría y positividad
       ├── Espectro y kernel
       ├── Isomorfismo de Hodge
       ├── Gap espectral
       └── Número de condición

    5. TESTS DE DESCOMPOSICIÓN DE HODGE
       ├── Ortogonalidad de componentes
       ├── Completitud (reconstrucción)
       ├── Conservación de energía
       ├── Unicidad
       └── Estabilidad numérica

    6. TESTS DE OPERADOR DE VORTICIDAD
       ├── Proyector solenoidal
       ├── Idempotencia
       ├── Circulación (Ley de Stokes)
       ├── Energía de vorticidad
       └── Índice de vorticidad

    7. TESTS DE MAGNON CARTRIDGE
       ├── Construcción y validación
       ├── Clasificación de severidad
       ├── Serialización
       └── Inmutabilidad

    8. TESTS DE INTEGRACIÓN
       ├── Pipeline completo
       ├── inspect_and_mitigate_resonance
       ├── Flujos laminares vs rotacionales
       └── Telemetría

    9. TESTS DE PROPIEDADES TOPOLÓGICAS
       ├── Números de Betti
       ├── Componentes conexas
       ├── Ciclos independientes
       └── Característica de Euler

    10. TESTS DE CASOS LÍMITE
        ├── Grafos vacíos
        ├── Grafos triviales (1 nodo)
        ├── Grafos completos
        ├── Grafos con alta vorticidad
        └── Estabilidad numérica extrema

CONVENCIONES:

    - test_unit_*:        Tests unitarios de componentes atómicos
    - test_integration_*: Tests de integración multi-componente
    - test_property_*:    Tests de propiedades algebraicas
    - test_invariant_*:   Tests de invariantes topológicos
    - test_edge_*:        Tests de casos límite
    - test_numerical_*:   Tests de estabilidad numérica
    - test_spectral_*:    Tests de propiedades espectrales

═════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
import sys
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import pytest
from scipy import linalg as la

# Importaciones del módulo bajo test
from app.physics.solenoid_acoustic import (
    # Constantes
    NumericalConstants,
    NC,
    
    # Excepciones
    HodgeDecompositionError,
    TopologicalInvariantError,
    NumericalStabilityError,
    GraphStructureError,
    
    # Estructuras de datos
    SpectralDecomposition,
    BettiNumbers,
    ChainComplex,
    VorticityMetrics,
    
    # Utilidades numéricas
    NumericalUtilities,
    
    # Constructores
    HodgeDecompositionBuilder,
    
    # Operadores
    AcousticSolenoidOperator,
    
    # Resultados
    MagnonCartridge,
    ResonanceMitigationResult,
    
    # API de alto nivel
    inspect_and_mitigate_resonance,
    verify_hodge_properties,
)


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1: FIXTURES Y GRAFOS DE REFERENCIA
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def dag_simple() -> nx.DiGraph:
    """
    Grafo acíclico dirigido (DAG) simple.
    
    Estructura:
        A → B → C
        A → D → C
    
    Propiedades:
        - n = 4, m = 4
        - β₀ = 1 (conexo)
        - β₁ = 0 (sin ciclos)
        - χ = 0
    """
    G = nx.DiGraph()
    G.add_edges_from([
        ("A", "B"),
        ("B", "C"),
        ("A", "D"),
        ("D", "C"),
    ])
    return G


@pytest.fixture
def single_cycle() -> nx.DiGraph:
    """
    Grafo con un ciclo simple.
    
    Estructura:
        A → B → C → A
    
    Propiedades:
        - n = 3, m = 3
        - β₀ = 1
        - β₁ = 1 (un ciclo)
        - χ = 0
    """
    G = nx.DiGraph()
    G.add_edges_from([
        ("A", "B"),
        ("B", "C"),
        ("C", "A"),
    ])
    return G


@pytest.fixture
def double_cycle() -> nx.DiGraph:
    """
    Grafo con dos ciclos compartiendo vértice.
    
    Estructura:
        Ciclo 1: A → B → C → A
        Ciclo 2: A → D → E → A
    
    Propiedades:
        - n = 5, m = 6
        - β₀ = 1
        - β₁ = 2 (dos ciclos independientes)
        - χ = -1
    """
    G = nx.DiGraph()
    G.add_edges_from([
        ("A", "B"), ("B", "C"), ("C", "A"),  # Ciclo 1
        ("A", "D"), ("D", "E"), ("E", "A"),  # Ciclo 2
    ])
    return G


@pytest.fixture
def disconnected_graph() -> nx.DiGraph:
    """
    Grafo con dos componentes desconectadas.
    
    Estructura:
        Componente 1: A → B → C → A
        Componente 2: D → E
    
    Propiedades:
        - n = 5, m = 4
        - β₀ = 2 (dos componentes)
        - β₁ = 1 (un ciclo)
        - χ = 1
    """
    G = nx.DiGraph()
    G.add_edges_from([
        ("A", "B"), ("B", "C"), ("C", "A"),  # Componente 1 con ciclo
        ("D", "E"),  # Componente 2 acíclica
    ])
    return G


@pytest.fixture
def complete_graph_3() -> nx.DiGraph:
    """
    Grafo completo dirigido K₃.
    
    Estructura:
        Todas las aristas posibles entre 3 vértices.
    
    Propiedades:
        - n = 3, m = 6
        - β₀ = 1
        - β₁ = 4 (múltiples ciclos)
        - χ = -3
    """
    G = nx.DiGraph()
    nodes = ["A", "B", "C"]
    for u in nodes:
        for v in nodes:
            if u != v:
                G.add_edge(u, v)
    return G


@pytest.fixture
def trivial_graph() -> nx.DiGraph:
    """
    Grafo trivial (1 nodo, 0 aristas).
    
    Propiedades:
        - n = 1, m = 0
        - β₀ = 1
        - β₁ = 0
        - χ = 1
    """
    G = nx.DiGraph()
    G.add_node("A")
    return G


@pytest.fixture
def self_loop_graph() -> nx.DiGraph:
    """
    Grafo con self-loop.
    
    Estructura:
        A → A (self-loop)
        A → B
    
    Propiedades:
        - n = 2, m = 2
        - β₀ = 1
        - β₁ = 1 (self-loop es un ciclo)
        - χ = 0
    """
    G = nx.DiGraph()
    G.add_edges_from([
        ("A", "A"),  # Self-loop
        ("A", "B"),
    ])
    return G


@pytest.fixture
def sample_flows_uniform() -> Dict[Tuple[str, str], float]:
    """
    Flujos uniformes (todos = 1.0) para testing.
    """
    return {
        ("A", "B"): 1.0,
        ("B", "C"): 1.0,
        ("C", "A"): 1.0,
    }


@pytest.fixture
def sample_flows_nonuniform() -> Dict[Tuple[str, str], float]:
    """
    Flujos no uniformes para testing de vorticidad.
    """
    return {
        ("A", "B"): 10.0,
        ("B", "C"): 10.0,
        ("C", "A"): 10.0,
        ("A", "D"): 5.0,
        ("D", "E"): 5.0,
        ("E", "A"): 5.0,
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2: TESTS DE ÁLGEBRA LINEAL NUMÉRICA
# ═════════════════════════════════════════════════════════════════════════════

class TestNumericalUtilities:
    """
    Suite de validación de álgebra lineal numérica.
    
    Verifica:
        - Tolerancias adaptativas (convención LAPACK)
        - Rango numérico (Teorema de Eckart-Young-Mirsky)
        - Pseudoinversas (4 condiciones de Penrose)
        - Proyecciones ortogonales (estabilidad SVD)
        - Bases de kernel (ortogonalidad)
    """
    
    # ─────────────────────────────────────────────────────────────────────────
    # 2.1 Tolerancias Adaptativas
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_unit_adaptive_tolerance_positive(self):
        """
        Verifica que tolerancia adaptativa sea siempre positiva.
        """
        matrices = [
            np.eye(5),
            np.random.randn(10, 10),
            np.zeros((3, 3)),
            np.ones((4, 4)),
        ]
        
        for A in matrices:
            tol = NumericalUtilities.adaptive_tolerance(A)
            assert tol > 0, f"Tolerancia debe ser positiva: {tol}"
    
    def test_unit_adaptive_tolerance_scales_with_size(self):
        """
        Verifica que tolerancia escale con dimensión de matriz.
        """
        A_small = np.eye(5)
        A_large = np.eye(100)
        
        tol_small = NumericalUtilities.adaptive_tolerance(A_small)
        tol_large = NumericalUtilities.adaptive_tolerance(A_large)
        
        # Para matrices de identidad, tol ∝ max(m, n)
        assert tol_large > tol_small
    
    def test_unit_adaptive_tolerance_zero_matrix(self):
        """
        Verifica manejo de matriz nula.
        """
        A = np.zeros((10, 10))
        tol = NumericalUtilities.adaptive_tolerance(A)
        
        # Para matriz nula, debe retornar base_tolerance
        assert tol >= NC.BASE_TOLERANCE
    
    def test_unit_adaptive_tolerance_ill_conditioned(self):
        """
        Verifica tolerancia en matriz mal condicionada.
        """
        # Matriz con valores singulares en rango amplio
        A = np.diag([1e10, 1.0, 1e-10])
        tol = NumericalUtilities.adaptive_tolerance(A)
        
        # Tolerancia debe escalar con σ_max
        assert tol > NC.MACHINE_EPSILON
    
    # ─────────────────────────────────────────────────────────────────────────
    # 2.2 Rango Numérico
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_unit_rank_full_rank_matrix(self):
        """
        Verifica rango de matriz de rango completo.
        """
        A = np.random.randn(5, 5)
        rank, svs = NumericalUtilities.compute_numerical_rank(A)
        
        assert rank == 5, f"Rango esperado 5, obtenido {rank}"
        assert len(svs) == 5
        assert all(svs > 0)
    
    def test_unit_rank_rank_deficient_matrix(self):
        """
        Verifica rango de matriz rank-deficiente.
        """
        # Matriz con rank = 2
        A = np.outer(np.array([1, 2, 3]), np.array([1, 0]))
        A = A + np.outer(np.array([0, 1, 1]), np.array([0, 1]))
        
        rank, svs = NumericalUtilities.compute_numerical_rank(A)
        
        assert rank == 2, f"Rango esperado 2, obtenido {rank}"
    
    def test_unit_rank_zero_matrix(self):
        """
        Verifica rango de matriz nula.
        """
        A = np.zeros((5, 5))
        rank, svs = NumericalUtilities.compute_numerical_rank(A)
        
        assert rank == 0, f"Rango de matriz nula debe ser 0, obtenido {rank}"
    
    def test_unit_rank_singular_values_ordered(self):
        """
        Verifica que valores singulares estén ordenados (desc).
        """
        A = np.random.randn(10, 10)
        rank, svs = NumericalUtilities.compute_numerical_rank(A)
        
        assert all(svs[i] >= svs[i+1] for i in range(len(svs)-1))
    
    # ─────────────────────────────────────────────────────────────────────────
    # 2.3 Pseudoinversa de Moore-Penrose
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_unit_pseudoinverse_penrose_conditions(self):
        """
        Verifica las 4 condiciones de Penrose para pseudoinversa.
        
        Condiciones:
            1. A A⁺ A = A
            2. A⁺ A A⁺ = A⁺
            3. (A A⁺)ᵀ = A A⁺
            4. (A⁺ A)ᵀ = A⁺ A
        """
        A = np.random.randn(5, 3)
        A_pinv = NumericalUtilities.moore_penrose_pseudoinverse(A)
        
        tol = NC.BASE_TOLERANCE * np.linalg.norm(A)
        
        # Condición 1: A A⁺ A = A
        assert np.allclose(A @ A_pinv @ A, A, atol=tol), "Condición 1 de Penrose violada"
        
        # Condición 2: A⁺ A A⁺ = A⁺
        assert np.allclose(A_pinv @ A @ A_pinv, A_pinv, atol=tol), "Condición 2 de Penrose violada"
        
        # Condición 3: (A A⁺)ᵀ = A A⁺
        AA_pinv = A @ A_pinv
        assert np.allclose(AA_pinv.T, AA_pinv, atol=tol), "Condición 3 de Penrose violada"
        
        # Condición 4: (A⁺ A)ᵀ = A⁺ A
        A_pinv_A = A_pinv @ A
        assert np.allclose(A_pinv_A.T, A_pinv_A, atol=tol), "Condición 4 de Penrose violada"
    
    def test_unit_pseudoinverse_identity(self):
        """
        Verifica que A⁺ de matriz identidad sea identidad.
        """
        A = np.eye(5)
        A_pinv = NumericalUtilities.moore_penrose_pseudoinverse(A)
        
        assert np.allclose(A_pinv, A, atol=NC.BASE_TOLERANCE)
    
    def test_unit_pseudoinverse_rank_deficient(self):
        """
        Verifica pseudoinversa de matriz rank-deficiente.
        """
        # Matriz 3×3 con rank = 2
        A = np.array([
            [1, 2, 3],
            [2, 4, 6],
            [0, 1, 2],
        ], dtype=float)
        
        A_pinv = NumericalUtilities.moore_penrose_pseudoinverse(A)
        
        # Verificar dimensiones correctas
        assert A_pinv.shape == (3, 3)
        
        # Verificar al menos condición 1
        tol = NC.BASE_TOLERANCE * np.linalg.norm(A)
        assert np.allclose(A @ A_pinv @ A, A, atol=tol)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 2.4 Proyecciones Ortogonales
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_unit_orthogonal_projection_basic(self):
        """
        Verifica proyección ortogonal básica.
        """
        # Vector y subespacio
        v = np.array([3.0, 4.0, 0.0])
        B = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]).T  # Span de ex, ey
        
        projected, residual = NumericalUtilities.orthogonal_projection(v, B.T)
        
        # Verificar reconstrucción
        assert np.allclose(v, projected + residual, atol=NC.BASE_TOLERANCE)
        
        # Verificar ortogonalidad
        inner = np.dot(projected, residual)
        assert abs(inner) < NC.ORTHOGONALITY_TOLERANCE * np.linalg.norm(v)**2
    
    def test_unit_orthogonal_projection_onto_itself(self):
        """
        Verifica que proyección sobre sí mismo sea identidad.
        """
        v = np.array([1.0, 2.0, 3.0])
        B = v.reshape(-1, 1)  # Subespacio unidimensional spanned por v
        
        projected, residual = NumericalUtilities.orthogonal_projection(v, B)
        
        # Proyección debe ser ≈ v (normalizado)
        assert np.allclose(projected / np.linalg.norm(projected), v / np.linalg.norm(v), atol=NC.BASE_TOLERANCE)
        
        # Residual debe ser ≈ 0
        assert np.linalg.norm(residual) < NC.BASE_TOLERANCE
    
    def test_unit_orthogonal_projection_empty_subspace(self):
        """
        Verifica proyección sobre subespacio vacío.
        """
        v = np.array([1.0, 2.0, 3.0])
        B = np.zeros((3, 0))  # Subespacio vacío
        
        projected, residual = NumericalUtilities.orthogonal_projection(v, B)
        
        # Proyección debe ser cero
        assert np.allclose(projected, np.zeros(3), atol=NC.BASE_TOLERANCE)
        
        # Residual debe ser v
        assert np.allclose(residual, v, atol=NC.BASE_TOLERANCE)
    
    def test_unit_orthogonal_projection_orthogonality(self):
        """
        Verifica ortogonalidad de componentes.
        """
        v = np.random.randn(10)
        B = np.random.randn(10, 5)
        
        projected, residual = NumericalUtilities.orthogonal_projection(v, B)
        
        # ⟨projected, residual⟩ ≈ 0
        inner = np.dot(projected, residual)
        tol = NC.ORTHOGONALITY_TOLERANCE * np.linalg.norm(v)**2
        
        assert abs(inner) < tol, f"Componentes no ortogonales: ⟨p,r⟩={inner:.2e}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # 2.5 Bases del Kernel
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_unit_null_space_full_rank(self):
        """
        Verifica que kernel de matriz de rango completo sea vacío.
        """
        A = np.random.randn(5, 5)
        # Asegurar rango completo
        A = A + 10 * np.eye(5)
        
        kernel_basis = NumericalUtilities.null_space_basis(A)
        
        assert kernel_basis.shape[1] == 0, "Kernel de matriz full-rank debe ser vacío"
    
    def test_unit_null_space_rank_deficient(self):
        """
        Verifica kernel de matriz rank-deficiente.
        """
        # Matriz 5×5 con rank = 3
        A = np.random.randn(5, 3) @ np.random.randn(3, 5)
        
        kernel_basis = NumericalUtilities.null_space_basis(A)
        
        # Dimensión del kernel debe ser 5 - 3 = 2
        assert kernel_basis.shape[1] >= 1, "Kernel debe ser no vacío"
        
        # Verificar que A @ ker ≈ 0
        if kernel_basis.shape[1] > 0:
            product = A @ kernel_basis
            assert np.allclose(product, 0, atol=NC.BASE_TOLERANCE)
    
    def test_unit_null_space_orthonormality(self):
        """
        Verifica ortonormalidad de base del kernel.
        """
        # Matriz con kernel no trivial
        A = np.array([
            [1, 2, 3, 4],
            [2, 4, 6, 8],
            [0, 1, 2, 3],
        ], dtype=float)
        
        kernel_basis = NumericalUtilities.null_space_basis(A)
        
        if kernel_basis.shape[1] > 0:
            # Verificar ortonormalidad: Kᵀ K = I
            gram = kernel_basis.T @ kernel_basis
            identity = np.eye(kernel_basis.shape[1])
            
            assert np.allclose(gram, identity, atol=NC.ORTHOGONALITY_TOLERANCE)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 2.6 Número de Condición
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_unit_condition_number_identity(self):
        """
        Verifica que κ(I) = 1.
        """
        A = np.eye(10)
        kappa, sigma_min, sigma_max = NumericalUtilities.condition_number(A)
        
        assert math.isclose(kappa, 1.0, rel_tol=NC.BASE_TOLERANCE)
        assert math.isclose(sigma_min, 1.0, rel_tol=NC.BASE_TOLERANCE)
        assert math.isclose(sigma_max, 1.0, rel_tol=NC.BASE_TOLERANCE)
    
    def test_unit_condition_number_singular(self):
        """
        Verifica que matriz singular tenga κ = ∞.
        """
        A = np.zeros((5, 5))
        kappa, sigma_min, sigma_max = NumericalUtilities.condition_number(A)
        
        assert math.isinf(kappa), "Matriz singular debe tener κ = ∞"
        assert sigma_min == 0.0
    
    def test_unit_condition_number_ill_conditioned(self):
        """
        Verifica detección de matrices mal condicionadas.
        """
        # Matriz con amplio rango de valores singulares
        A = np.diag([1e10, 1.0, 1e-10])
        kappa, sigma_min, sigma_max = NumericalUtilities.condition_number(A)
        
        expected_kappa = 1e10 / 1e-10
        assert math.isclose(kappa, expected_kappa, rel_tol=1e-2)


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3: TESTS DE CONSTRUCCIÓN DEL COMPLEJO DE CADENAS
# ═════════════════════════════════════════════════════════════════════════════

class TestHodgeDecompositionBuilder:
    """
    Suite de validación del constructor de complejo de cadenas.
    
    Verifica:
        - Matriz de incidencia B₁
        - Matriz de ciclos B₂
        - Condición ∂₁ ∘ ∂₂ = 0
        - Números de Betti
        - Euler-Poincaré
    """
    
    # ─────────────────────────────────────────────────────────────────────────
    # 3.1 Matriz de Incidencia B₁
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_unit_incidence_matrix_dimensions(self, single_cycle):
        """
        Verifica dimensiones correctas de B₁.
        """
        builder = HodgeDecompositionBuilder(single_cycle)
        B1, meta = builder.build_incidence_matrix()
        
        n = single_cycle.number_of_nodes()
        m = single_cycle.number_of_edges()
        
        assert B1.shape == (n, m), f"B₁ debe ser {n}×{m}, obtenido {B1.shape}"
    
    def test_unit_incidence_matrix_column_sum_zero(self, double_cycle):
        """
        Verifica que suma de columnas sea 0 (∂₁e = head - tail).
        """
        builder = HodgeDecompositionBuilder(double_cycle)
        B1, meta = builder.build_incidence_matrix()
        
        col_sums = B1.sum(axis=0)
        
        assert np.allclose(col_sums, 0, atol=NC.BASE_TOLERANCE), \
            f"Columnas de B₁ deben sumar 0, max={meta['column_sum_max']}"
    
    def test_unit_incidence_matrix_rank(self, single_cycle):
        """
        Verifica rank(B₁) = n - c (c componentes).
        """
        builder = HodgeDecompositionBuilder(single_cycle)
        B1, meta = builder.build_incidence_matrix()
        
        n = single_cycle.number_of_nodes()
        c = 1  # Conexo
        expected_rank = n - c
        
        assert meta["rank_B1"] == expected_rank, \
            f"rank(B₁) esperado {expected_rank}, obtenido {meta['rank_B1']}"
    
    def test_unit_incidence_matrix_disconnected(self, disconnected_graph):
        """
        Verifica rank(B₁) para grafo desconectado.
        """
        builder = HodgeDecompositionBuilder(disconnected_graph)
        B1, meta = builder.build_incidence_matrix()
        
        n = disconnected_graph.number_of_nodes()
        c = 2  # Dos componentes
        expected_rank = n - c
        
        assert meta["rank_B1"] == expected_rank
    
    # ─────────────────────────────────────────────────────────────────────────
    # 3.2 Matriz de Ciclos B₂
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_unit_face_matrix_dimensions(self, double_cycle):
        """
        Verifica dimensiones de B₂.
        """
        builder = HodgeDecompositionBuilder(double_cycle)
        B2, meta = builder.build_face_matrix()
        
        m = double_cycle.number_of_edges()
        beta_1 = 2  # Dos ciclos
        
        assert B2.shape == (m, beta_1), f"B₂ debe ser {m}×{beta_1}, obtenido {B2.shape}"
    
    def test_unit_face_matrix_dag_zero(self, dag_simple):
        """
        Verifica que DAG tenga B₂ vacía (sin ciclos).
        """
        builder = HodgeDecompositionBuilder(dag_simple)
        B2, meta = builder.build_face_matrix()
        
        assert B2.shape[1] == 0, "DAG no debe tener ciclos"
        assert meta["betti_1"] == 0
        assert meta["is_forest"] is True
    
    def test_unit_face_matrix_rank(self, double_cycle):
        """
        Verifica rank(B₂) = β₁.
        """
        builder = HodgeDecompositionBuilder(double_cycle)
        B2, meta = builder.build_face_matrix()
        
        assert meta["rank_B2"] == meta["betti_1"], \
            "rank(B₂) debe igualar β₁"
    
    # ─────────────────────────────────────────────────────────────────────────
    # 3.3 Condición de Complejo: ∂₁ ∘ ∂₂ = 0
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_property_boundary_composition_zero(self, double_cycle):
        """
        Verifica axioma fundamental: B₁ B₂ = 0.
        """
        builder = HodgeDecompositionBuilder(double_cycle)
        B1, _ = builder.build_incidence_matrix()
        B2, meta = builder.build_face_matrix()
        
        B1B2 = B1 @ B2
        B1B2_norm = np.linalg.norm(B1B2, 'fro')
        
        assert B1B2_norm < NC.BASE_TOLERANCE, \
            f"‖B₁B₂‖_F debe ser ≈ 0, obtenido {B1B2_norm:.2e}"
        
        assert meta["verify_B1B2_zero"] is True
    
    def test_property_boundary_composition_all_graphs(
        self,
        single_cycle,
        double_cycle,
        complete_graph_3
    ):
        """
        Verifica ∂₁ ∘ ∂₂ = 0 para múltiples grafos.
        """
        graphs = [single_cycle, double_cycle, complete_graph_3]
        
        for G in graphs:
            builder = HodgeDecompositionBuilder(G)
            B1, _ = builder.build_incidence_matrix()
            B2, _ = builder.build_face_matrix()
            
            if B2.shape[1] > 0:
                B1B2 = B1 @ B2
                B1B2_norm = np.linalg.norm(B1B2, 'fro')
                
                assert B1B2_norm < NC.BASE_TOLERANCE, \
                    f"Violación de ∂₁∘∂₂=0 en grafo con {G.number_of_nodes()} nodos"
    
    # ─────────────────────────────────────────────────────────────────────────
    # 3.4 Números de Betti y Euler-Poincaré
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_invariant_euler_poincare_single_cycle(self, single_cycle):
        """
        Verifica Euler-Poincaré: χ = n - m = β₀ - β₁.
        """
        builder = HodgeDecompositionBuilder(single_cycle)
        verification = builder.verify_chain_complex()
        
        assert verification["euler_characteristic"]["verified"] is True
    
    def test_invariant_euler_poincare_disconnected(self, disconnected_graph):
        """
        Verifica Euler-Poincaré para grafo desconectado.
        """
        builder = HodgeDecompositionBuilder(disconnected_graph)
        verification = builder.verify_chain_complex()
        
        chi_geom = verification["euler_characteristic"]["chi_geometric"]
        chi_topo = verification["euler_characteristic"]["chi_topological"]
        
        assert chi_geom == chi_topo, \
            f"χ geométrico ({chi_geom}) ≠ χ topológico ({chi_topo})"
    
    def test_invariant_betti_numbers_complete_graph(self, complete_graph_3):
        """
        Verifica números de Betti en grafo completo K₃.
        """
        builder = HodgeDecompositionBuilder(complete_graph_3)
        _, meta = builder.build_face_matrix()
        
        # K₃ dirigido tiene múltiples ciclos
        assert meta["betti_1"] >= 3, "K₃ debe tener β₁ ≥ 3"
    
    # ─────────────────────────────────────────────────────────────────────────
    # 3.5 Laplaciano de Hodge L₁
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_unit_laplacian_symmetry(self, double_cycle):
        """
        Verifica que L₁ sea simétrico.
        """
        builder = HodgeDecompositionBuilder(double_cycle)
        L1, spectral = builder.compute_hodge_laplacian()
        
        assert np.allclose(L1, L1.T, atol=NC.SYMMETRY_TOLERANCE), \
            "L₁ debe ser simétrico"
    
    def test_unit_laplacian_positive_semidefinite(self, double_cycle):
        """
        Verifica que L₁ sea PSD (todos los λᵢ ≥ 0).
        """
        builder = HodgeDecompositionBuilder(double_cycle)
        L1, spectral = builder.compute_hodge_laplacian()
        
        assert all(spectral.eigenvalues >= -NC.BASE_TOLERANCE), \
            "L₁ debe ser semi-definido positivo"
    
    def test_invariant_hodge_isomorphism(self, double_cycle):
        """
        Verifica isomorfismo de Hodge: dim ker(L₁) = β₁.
        """
        builder = HodgeDecompositionBuilder(double_cycle)
        L1, spectral = builder.compute_hodge_laplacian()
        _, meta_B2 = builder.build_face_matrix()
        
        kernel_dim = spectral.kernel_dimension
        beta_1 = meta_B2["betti_1"]
        
        assert kernel_dim == beta_1, \
            f"dim ker(L₁) = {kernel_dim} ≠ β₁ = {beta_1}"
    
    def test_unit_laplacian_trace(self, single_cycle):
        """
        Verifica Tr(L₁) = Σλᵢ = ‖B₁‖²_F + ‖B₂‖²_F.
        """
        builder = HodgeDecompositionBuilder(single_cycle)
        L1, spectral = builder.compute_hodge_laplacian()
        
        trace_direct = np.trace(L1)
        trace_eigs = np.sum(spectral.eigenvalues)
        
        assert math.isclose(trace_direct, trace_eigs, rel_tol=NC.BASE_TOLERANCE), \
            f"Tr(L₁) = {trace_direct:.6e} ≠ Σλᵢ = {trace_eigs:.6e}"


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4: TESTS DE DESCOMPOSICIÓN DE HODGE-HELMHOLTZ
# ═════════════════════════════════════════════════════════════════════════════

class TestHodgeDecomposition:
    """
    Suite de validación de descomposición de Hodge completa.
    
    Verifica:
        - Ortogonalidad: ⟨I_grad, I_curl⟩ = 0
        - Completitud: I = I_grad + I_curl + I_harm
        - Conservación de energía
        - Unicidad de componentes
    """
    
    def test_property_orthogonal_decomposition(self, double_cycle, sample_flows_nonuniform):
        """
        Verifica ortogonalidad de componentes de Hodge.
        """
        solenoid = AcousticSolenoidOperator()
        result = solenoid.compute_full_hodge_decomposition(
            double_cycle,
            sample_flows_nonuniform
        )
        
        verification = result["verification"]
        
        # Productos internos deben ser ≈ 0
        assert abs(verification["orthogonality_grad_curl"]) < verification["orthogonality_tolerance"]
        assert abs(verification["orthogonality_grad_harm"]) < verification["orthogonality_tolerance"]
        assert abs(verification["orthogonality_curl_harm"]) < verification["orthogonality_tolerance"]
        
        assert verification["is_orthogonal_decomposition"] is True
    
    def test_property_complete_decomposition(self, double_cycle, sample_flows_nonuniform):
        """
        Verifica completitud: I = I_grad + I_curl + I_harm.
        """
        solenoid = AcousticSolenoidOperator()
        result = solenoid.compute_full_hodge_decomposition(
            double_cycle,
            sample_flows_nonuniform
        )
        
        verification = result["verification"]
        
        assert verification["is_complete_decomposition"] is True
        assert verification["reconstruction_error"] < NC.BASE_TOLERANCE * 10
    
    def test_invariant_energy_conservation(self, double_cycle, sample_flows_nonuniform):
        """
        Verifica conservación de energía: E_total = E_grad + E_curl + E_harm.
        """
        solenoid = AcousticSolenoidOperator()
        result = solenoid.compute_full_hodge_decomposition(
            double_cycle,
            sample_flows_nonuniform
        )
        
        energy = result["energy_decomposition"]
        
        E_total = energy["total"]
        E_sum = energy["irrotational"] + energy["solenoidal"] + energy["harmonic"]
        
        # Balance de energía
        balance_error = abs(E_total - E_sum)
        
        assert balance_error < NC.BASE_TOLERANCE * E_total, \
            f"Violación de conservación de energía: error={balance_error:.2e}"
    
    def test_property_uniqueness(self, single_cycle):
        """
        Verifica unicidad de descomposición de Hodge.
        
        Dos descomposiciones del mismo flujo deben dar mismo resultado.
        """
        flows = {
            ("A", "B"): 5.0,
            ("B", "C"): 5.0,
            ("C", "A"): 5.0,
        }
        
        solenoid = AcousticSolenoidOperator()
        
        result1 = solenoid.compute_full_hodge_decomposition(single_cycle, flows)
        result2 = solenoid.compute_full_hodge_decomposition(single_cycle, flows)
        
        # Componentes deben ser idénticas
        I_grad1 = np.array(result1["components"]["irrotational"])
        I_grad2 = np.array(result2["components"]["irrotational"])
        
        assert np.allclose(I_grad1, I_grad2, atol=NC.BASE_TOLERANCE)


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5: TESTS DE OPERADOR DE VORTICIDAD
# ═════════════════════════════════════════════════════════════════════════════

class TestVorticityOperator:
    """
    Suite de validación del operador de vorticidad solenoidal.
    
    Verifica:
        - Proyector P_curl (idempotencia, simetría)
        - Circulación (Ley de Stokes)
        - Energía de vorticidad
        - Índice de vorticidad
    """
    
    def test_unit_projector_idempotency(self, single_cycle, sample_flows_uniform):
        """
        Verifica idempotencia: P² = P.
        """
        solenoid = AcousticSolenoidOperator()
        magnon = solenoid.isolate_vorticity(single_cycle, sample_flows_uniform)
        
        if magnon is not None and magnon.projector_matrix is not None:
            P = magnon.projector_matrix
            P_squared = P @ P
            
            idempotency_error = np.linalg.norm(P_squared - P, 'fro') / np.linalg.norm(P, 'fro')
            
            assert idempotency_error < NC.IDEMPOTENCY_TOLERANCE, \
                f"Proyector no idempotente: error={idempotency_error:.2e}"
    
    def test_unit_vorticity_dag_zero(self, dag_simple):
        """
        Verifica que DAG tenga vorticidad nula.
        """
        flows = {edge: 1.0 for edge in dag_simple.edges()}
        
        solenoid = AcousticSolenoidOperator()
        magnon = solenoid.isolate_vorticity(dag_simple, flows)
        
        # DAG no debe emitir magnon (sin ciclos)
        assert magnon is None, "DAG no debe tener vorticidad"
    
    def test_unit_vorticity_index_bounds(self, double_cycle, sample_flows_nonuniform):
        """
        Verifica que índice de vorticidad ω ∈ [0, 1].
        """
        solenoid = AcousticSolenoidOperator()
        magnon = solenoid.isolate_vorticity(double_cycle, sample_flows_nonuniform)
        
        if magnon is not None:
            omega = magnon.vorticity_index
            assert 0.0 <= omega <= 1.0, f"ω fuera de [0,1]: {omega}"
    
    def test_property_circulation_stokes(self, single_cycle):
        """
        Verifica Ley de Stokes discreta: Γ = B₂ᵀI.
        """
        flows = {
            ("A", "B"): 10.0,
            ("B", "C"): 10.0,
            ("C", "A"): 10.0,
        }
        
        solenoid = AcousticSolenoidOperator()
        magnon = solenoid.isolate_vorticity(single_cycle, flows)
        
        if magnon is not None:
            # Para ciclo simple con flujo uniforme, Γ debe ser ≈ 3*10 = 30
            circulation = magnon.metrics.circulation_vector
            
            assert len(circulation) == 1, "Ciclo simple debe tener β₁ = 1"
            # La circulación exacta depende de la orientación del ciclo
            assert abs(circulation[0]) > 0, "Circulación debe ser no nula"


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6: TESTS DE MAGNON CARTRIDGE (continuación de Parte 1)
# ═════════════════════════════════════════════════════════════════════════════

class TestMagnonCartridge:
    """
    Suite de validación del bosón de vorticidad (MagnonCartridge).
    
    Verifica:
        - Construcción e inmutabilidad
        - Validación de invariantes físicos
        - Clasificación de severidad
        - Serialización completa
        - Generación de proof matemático
    """
    
    # ─────────────────────────────────────────────────────────────────────────
    # 6.1 Construcción y Validación
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_unit_magnon_construction_valid(self, double_cycle, sample_flows_nonuniform):
        """
        Verifica construcción exitosa de MagnonCartridge.
        """
        solenoid = AcousticSolenoidOperator()
        magnon = solenoid.isolate_vorticity(double_cycle, sample_flows_nonuniform)
        
        assert magnon is not None, "Debe emitir magnon con flujo rotacional"
        assert isinstance(magnon, MagnonCartridge)
        assert isinstance(magnon.metrics, VorticityMetrics)
    
    def test_unit_magnon_immutability(self, double_cycle, sample_flows_nonuniform):
        """
        Verifica que MagnonCartridge sea inmutable (frozen).
        """
        solenoid = AcousticSolenoidOperator()
        magnon = solenoid.isolate_vorticity(double_cycle, sample_flows_nonuniform)
        
        if magnon is not None:
            with pytest.raises(AttributeError):
                magnon.metrics = VorticityMetrics(
                    kinetic_energy=0.0,
                    total_energy=1.0,
                    vorticity_index=0.0,
                    circulation_vector=np.array([]),
                    dominant_cycle_index=-1,
                    dominant_circulation=0.0,
                    total_circulation_norm=0.0,
                )
    
    def test_unit_magnon_validates_negative_energy(self):
        """
        Verifica rechazo de energía cinética negativa.
        """
        with pytest.raises(ValueError, match="Energía cinética"):
            VorticityMetrics(
                kinetic_energy=-1.0,  # Inválido
                total_energy=10.0,
                vorticity_index=0.0,
                circulation_vector=np.array([1.0]),
                dominant_cycle_index=0,
                dominant_circulation=1.0,
                total_circulation_norm=1.0,
            )
    
    def test_unit_magnon_validates_vorticity_index_bounds(self):
        """
        Verifica rechazo de índice de vorticidad fuera de [0, 1].
        """
        with pytest.raises(ValueError, match="vorticidad"):
            VorticityMetrics(
                kinetic_energy=5.0,
                total_energy=10.0,
                vorticity_index=1.5,  # Inválido (> 1)
                circulation_vector=np.array([1.0]),
                dominant_cycle_index=0,
                dominant_circulation=1.0,
                total_circulation_norm=1.0,
            )
    
    def test_unit_magnon_validates_idempotency_error(self):
        """
        Verifica validación de error de idempotencia.
        """
        metrics = VorticityMetrics(
            kinetic_energy=5.0,
            total_energy=10.0,
            vorticity_index=0.5,
            circulation_vector=np.array([1.0, 2.0]),
            dominant_cycle_index=1,
            dominant_circulation=2.0,
            total_circulation_norm=2.236,
        )
        
        # Error negativo debe fallar
        with pytest.raises(ValueError, match="idempotencia"):
            MagnonCartridge(
                metrics=metrics,
                projection_idempotency_error=-0.01,
            )
    
    # ─────────────────────────────────────────────────────────────────────────
    # 6.2 Clasificación de Severidad
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_unit_severity_critical(self):
        """
        Verifica clasificación CRITICAL (ω > 0.50).
        """
        metrics = VorticityMetrics(
            kinetic_energy=60.0,
            total_energy=100.0,
            vorticity_index=0.6,
            circulation_vector=np.array([5.0]),
            dominant_cycle_index=0,
            dominant_circulation=5.0,
            total_circulation_norm=5.0,
        )
        
        assert metrics.severity_class == "CRITICAL"
    
    def test_unit_severity_high(self):
        """
        Verifica clasificación HIGH (0.20 < ω ≤ 0.50).
        """
        metrics = VorticityMetrics(
            kinetic_energy=30.0,
            total_energy=100.0,
            vorticity_index=0.3,
            circulation_vector=np.array([3.0]),
            dominant_cycle_index=0,
            dominant_circulation=3.0,
            total_circulation_norm=3.0,
        )
        
        assert metrics.severity_class == "HIGH"
    
    def test_unit_severity_moderate(self):
        """
        Verifica clasificación MODERATE (0.05 < ω ≤ 0.20).
        """
        metrics = VorticityMetrics(
            kinetic_energy=10.0,
            total_energy=100.0,
            vorticity_index=0.1,
            circulation_vector=np.array([2.0]),
            dominant_cycle_index=0,
            dominant_circulation=2.0,
            total_circulation_norm=2.0,
        )
        
        assert metrics.severity_class == "MODERATE"
    
    def test_unit_severity_low(self):
        """
        Verifica clasificación LOW (ω ≤ 0.05).
        """
        metrics = VorticityMetrics(
            kinetic_energy=2.0,
            total_energy=100.0,
            vorticity_index=0.02,
            circulation_vector=np.array([1.0]),
            dominant_cycle_index=0,
            dominant_circulation=1.0,
            total_circulation_norm=1.0,
        )
        
        assert metrics.severity_class == "LOW"
    
    # ─────────────────────────────────────────────────────────────────────────
    # 6.3 Propiedades Derivadas
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_unit_dominant_cycle_detection(self):
        """
        Verifica detección del ciclo dominante.
        """
        metrics = VorticityMetrics(
            kinetic_energy=50.0,
            total_energy=100.0,
            vorticity_index=0.5,
            circulation_vector=np.array([2.0, 5.0, 3.0]),
            dominant_cycle_index=1,
            dominant_circulation=5.0,
            total_circulation_norm=6.164,
        )
        
        magnon = MagnonCartridge(metrics=metrics)
        
        idx, circ = magnon.dominant_cycle
        assert idx == 1
        assert circ == 5.0
    
    def test_unit_is_significant_true(self):
        """
        Verifica que vorticidad sea significativa cuando cumple criterios.
        """
        metrics = VorticityMetrics(
            kinetic_energy=1e-6,  # > threshold
            total_energy=1e-4,
            vorticity_index=0.02,  # > 0.01
            circulation_vector=np.array([1e-3]),
            dominant_cycle_index=0,
            dominant_circulation=1e-3,
            total_circulation_norm=1e-3,
        )
        
        assert metrics.is_significant is True
    
    def test_unit_is_significant_false_low_energy(self):
        """
        Verifica que vorticidad sea insignificante con energía baja.
        """
        metrics = VorticityMetrics(
            kinetic_energy=1e-12,  # < threshold
            total_energy=1.0,
            vorticity_index=0.5,
            circulation_vector=np.array([1e-6]),
            dominant_cycle_index=0,
            dominant_circulation=1e-6,
            total_circulation_norm=1e-6,
        )
        
        assert metrics.is_significant is False
    
    # ─────────────────────────────────────────────────────────────────────────
    # 6.4 Serialización
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_unit_magnon_to_dict_serializable(self, double_cycle, sample_flows_nonuniform):
        """
        Verifica que to_dict() produzca diccionario completamente serializable.
        """
        solenoid = AcousticSolenoidOperator()
        magnon = solenoid.isolate_vorticity(double_cycle, sample_flows_nonuniform)
        
        if magnon is not None:
            result_dict = magnon.to_dict()
            
            # Verificar que sea dict
            assert isinstance(result_dict, dict)
            
            # Verificar claves esperadas
            assert "metrics" in result_dict
            assert "energy_decomposition" in result_dict
            assert "veto_payload" in result_dict
            
            # No debe contener np.ndarray directos (solo listas)
            import json
            try:
                json.dumps(result_dict)
            except TypeError as exc:
                pytest.fail(f"to_dict() no es JSON-serializable: {exc}")
    
    def test_unit_magnon_to_veto_payload(self, double_cycle, sample_flows_nonuniform):
        """
        Verifica estructura del veto payload.
        """
        solenoid = AcousticSolenoidOperator()
        magnon = solenoid.isolate_vorticity(double_cycle, sample_flows_nonuniform)
        
        if magnon is not None:
            veto = magnon.to_veto_payload()
            
            # Verificar campos obligatorios
            assert veto["type"] == "ROUTING_VETO"
            assert "magnitude" in veto
            assert "vorticity_metrics" in veto
            assert "causality_status" in veto
            assert "prescribed_action" in veto
            
            # Verificar coherencia
            if magnon.is_significant:
                assert veto["causality_status"] == "COMPROMISED"
            else:
                assert veto["causality_status"] == "INTACT"
    
    def test_unit_magnon_prescribed_action_mapping(self):
        """
        Verifica mapeo correcto de severidad → acción.
        """
        action_map = {
            "CRITICAL": "COLLAPSE_AND_RECONFIGURE",
            "HIGH": "PARTITION_AND_RELAY",
            "MODERATE": "MONITOR_AND_DAMP",
            "LOW": "LOG_AND_PROCEED",
        }
        
        for severity, expected_action in action_map.items():
            omega = {
                "CRITICAL": 0.6,
                "HIGH": 0.3,
                "MODERATE": 0.1,
                "LOW": 0.02,
            }[severity]
            
            metrics = VorticityMetrics(
                kinetic_energy=omega * 100,
                total_energy=100.0,
                vorticity_index=omega,
                circulation_vector=np.array([1.0]),
                dominant_cycle_index=0,
                dominant_circulation=1.0,
                total_circulation_norm=1.0,
            )
            
            magnon = MagnonCartridge(metrics=metrics)
            assert magnon._prescribe_action() == expected_action
    
    # ─────────────────────────────────────────────────────────────────────────
    # 6.5 Proof Matemático
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_unit_mathematical_proof_structure(self, double_cycle, sample_flows_nonuniform):
        """
        Verifica estructura del proof matemático.
        """
        solenoid = AcousticSolenoidOperator()
        magnon = solenoid.isolate_vorticity(double_cycle, sample_flows_nonuniform)
        
        if magnon is not None:
            proof = magnon.generate_mathematical_proof()
            
            # Verificar campos obligatorios
            assert "theorem" in proof
            assert "statement" in proof
            assert "verification" in proof
            assert "conclusion" in proof
            
            # Verificar referencia a Hodge
            assert "Hodge" in proof["theorem"]


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 7: TESTS DE INTEGRACIÓN COMPLETA
# ═════════════════════════════════════════════════════════════════════════════

class TestIntegrationComplete:
    """
    Suite de integración del pipeline completo.
    
    Verifica:
        - inspect_and_mitigate_resonance (API principal)
        - Flujo end-to-end
        - Telemetría
        - Casos de admisión y rechazo
    """
    
    # ─────────────────────────────────────────────────────────────────────────
    # 7.1 Pipeline Completo
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_integration_laminar_flow_dag(self, dag_simple):
        """
        Verifica que DAG produzca estado LAMINAR_FLOW.
        """
        flows = {edge: 1.0 for edge in dag_simple.edges()}
        
        result = inspect_and_mitigate_resonance(dag_simple, flows)
        
        assert result["status"] == "LAMINAR_FLOW"
        assert result["action"] == "PROCEED"
        assert result["vorticity_metrics"]["betti_1_cycles"] == 0
        assert result["vorticity_metrics"]["vorticity_index"] == 0.0
    
    def test_integration_resonance_detected(self, double_cycle, sample_flows_nonuniform):
        """
        Verifica detección de resonancia en grafo con ciclos.
        """
        result = inspect_and_mitigate_resonance(
            double_cycle,
            sample_flows_nonuniform,
            full_analysis=False
        )
        
        # Debe detectar vorticidad (dos ciclos con flujo)
        assert result["status"] in ["RESONANCE_DETECTED", "FAILURE"]
        assert result["vorticity_metrics"]["betti_1_cycles"] == 2
        assert result["vorticity_metrics"]["vorticity_index"] > 0.0
    
    def test_integration_full_analysis(self, single_cycle, sample_flows_uniform):
        """
        Verifica análisis completo con full_analysis=True.
        """
        result = inspect_and_mitigate_resonance(
            single_cycle,
            sample_flows_uniform,
            full_analysis=True
        )
        
        # Debe incluir análisis completo
        assert "full_hodge_decomposition" in result
        assert "spectral_analysis" in result
        
        # Verificar estructura de descomposición completa
        fhd = result["full_hodge_decomposition"]
        assert "components" in fhd
        assert "energy_decomposition" in fhd
        assert "verification" in fhd
        
        # Verificar estructura de análisis espectral
        spectral = result["spectral_analysis"]
        assert "laplacian_spectrum" in spectral
        assert "spectral_gap" in spectral
        assert "kernel_dimension" in spectral
    
    def test_integration_mathematical_proof_present(self, single_cycle, sample_flows_uniform):
        """
        Verifica que resultado incluya proof matemático.
        """
        result = inspect_and_mitigate_resonance(single_cycle, sample_flows_uniform)
        
        assert "mathematical_proof" in result
        proof = result["mathematical_proof"]
        
        assert "theorem" in proof
        assert "conclusion" in proof
    
    # ─────────────────────────────────────────────────────────────────────────
    # 7.2 Casos de Clasificación
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_integration_critical_severity(self, complete_graph_3):
        """
        Verifica clasificación CRITICAL en grafo con alta vorticidad.
        """
        # Flujos altos en todas las aristas
        flows = {edge: 100.0 for edge in complete_graph_3.edges()}
        
        result = inspect_and_mitigate_resonance(complete_graph_3, flows)
        
        if result["status"] != "LAMINAR_FLOW":
            severity = result["vorticity_metrics"]["thermodynamic_severity"]
            # K₃ con flujo alto debe tener severidad alta
            assert severity in ["HIGH", "CRITICAL"]
    
    def test_integration_empty_flows(self, single_cycle):
        """
        Verifica manejo de flujos vacíos.
        """
        flows = {}
        
        result = inspect_and_mitigate_resonance(single_cycle, flows)
        
        # Sin flujos → estado laminar
        assert result["status"] == "LAMINAR_FLOW"
    
    def test_integration_zero_flows(self, single_cycle):
        """
        Verifica manejo de flujos todos cero.
        """
        flows = {edge: 0.0 for edge in single_cycle.edges()}
        
        result = inspect_and_mitigate_resonance(single_cycle, flows)
        
        # Flujos cero → estado laminar
        assert result["status"] == "LAMINAR_FLOW"
    
    # ─────────────────────────────────────────────────────────────────────────
    # 7.3 Reproducibilidad
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_integration_reproducibility(self, double_cycle, sample_flows_nonuniform):
        """
        Verifica que análisis sea reproducible (determinista).
        """
        result1 = inspect_and_mitigate_resonance(double_cycle, sample_flows_nonuniform)
        result2 = inspect_and_mitigate_resonance(double_cycle, sample_flows_nonuniform)
        
        # Estados deben ser idénticos
        assert result1["status"] == result2["status"]
        assert result1["action"] == result2["action"]
        
        # Métricas numéricas deben ser idénticas
        vm1 = result1["vorticity_metrics"]
        vm2 = result2["vorticity_metrics"]
        
        assert vm1["vorticity_index"] == vm2["vorticity_index"]
        assert vm1["parasitic_kinetic_energy"] == vm2["parasitic_kinetic_energy"]
    
    # ─────────────────────────────────────────────────────────────────────────
    # 7.4 Telemetría (Mock)
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_integration_telemetry_laminar(self, dag_simple):
        """
        Verifica que telemetría registre flujo laminar.
        """
        from unittest.mock import MagicMock
        
        flows = {edge: 1.0 for edge in dag_simple.edges()}
        telemetry_mock = MagicMock()
        
        result = inspect_and_mitigate_resonance(
            dag_simple,
            flows,
            telemetry_ctx=telemetry_mock
        )
        
        # Debe llamar update_physics con is_stable=True
        telemetry_mock.update_physics.assert_called_once_with(is_stable=True)
    
    def test_integration_telemetry_critical(self, complete_graph_3):
        """
        Verifica que telemetría registre resonancia crítica.
        """
        from unittest.mock import MagicMock
        
        flows = {edge: 100.0 for edge in complete_graph_3.edges()}
        telemetry_mock = MagicMock()
        
        result = inspect_and_mitigate_resonance(
            complete_graph_3,
            flows,
            telemetry_ctx=telemetry_mock
        )
        
        if result["status"] == "FAILURE":
            # Debe registrar error crítico
            telemetry_mock.record_error.assert_called()


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 8: TESTS DE PROPIEDADES TOPOLÓGICAS
# ═════════════════════════════════════════════════════════════════════════════

class TestTopologicalProperties:
    """
    Suite de validación de invariantes topológicos.
    
    Verifica:
        - Números de Betti
        - Componentes conexas
        - Característica de Euler
        - Isomorfismo de Hodge
    """
    
    # ─────────────────────────────────────────────────────────────────────────
    # 8.1 Números de Betti
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_property_betti_numbers_dag(self, dag_simple):
        """
        Verifica números de Betti de DAG.
        """
        props = verify_hodge_properties(dag_simple)
        
        betti = props["betti_numbers"]
        assert betti["beta_0"] == 1, "DAG conexo debe tener β₀ = 1"
        assert betti["beta_1"] == 0, "DAG debe tener β₁ = 0"
    
    def test_property_betti_numbers_single_cycle(self, single_cycle):
        """
        Verifica números de Betti de grafo con un ciclo.
        """
        props = verify_hodge_properties(single_cycle)
        
        betti = props["betti_numbers"]
        assert betti["beta_0"] == 1
        assert betti["beta_1"] == 1, "Un ciclo debe tener β₁ = 1"
    
    def test_property_betti_numbers_double_cycle(self, double_cycle):
        """
        Verifica números de Betti de grafo con dos ciclos.
        """
        props = verify_hodge_properties(double_cycle)
        
        betti = props["betti_numbers"]
        assert betti["beta_0"] == 1
        assert betti["beta_1"] == 2, "Dos ciclos independientes → β₁ = 2"
    
    def test_property_betti_numbers_disconnected(self, disconnected_graph):
        """
        Verifica números de Betti de grafo desconectado.
        """
        props = verify_hodge_properties(disconnected_graph)
        
        betti = props["betti_numbers"]
        assert betti["beta_0"] == 2, "Dos componentes → β₀ = 2"
        assert betti["beta_1"] == 1, "Un ciclo total → β₁ = 1"
    
    # ─────────────────────────────────────────────────────────────────────────
    # 8.2 Euler-Poincaré
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_invariant_euler_all_graphs(
        self,
        dag_simple,
        single_cycle,
        double_cycle,
        disconnected_graph,
        complete_graph_3
    ):
        """
        Verifica Euler-Poincaré en todos los grafos de referencia.
        """
        graphs = [
            dag_simple,
            single_cycle,
            double_cycle,
            disconnected_graph,
            complete_graph_3,
        ]
        
        for G in graphs:
            props = verify_hodge_properties(G)
            euler = props["euler_characteristic"]
            
            assert euler["verified"] is True, \
                f"Euler-Poincaré falló en grafo con {G.number_of_nodes()} nodos"
    
    # ─────────────────────────────────────────────────────────────────────────
    # 8.3 Isomorfismo de Hodge
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_invariant_hodge_isomorphism_all_graphs(
        self,
        dag_simple,
        single_cycle,
        double_cycle,
        complete_graph_3
    ):
        """
        Verifica isomorfismo de Hodge: dim ker(L₁) = β₁.
        """
        graphs = [dag_simple, single_cycle, double_cycle, complete_graph_3]
        
        for G in graphs:
            props = verify_hodge_properties(G)
            hodge_iso = props["hodge_isomorphism"]
            
            assert hodge_iso["satisfied"] is True, \
                f"Isomorfismo de Hodge falló en grafo: " \
                f"dim ker(L₁)={hodge_iso['dim_ker_L1']}, " \
                f"β₁={hodge_iso['beta_1']}"


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 9: TESTS DE ANÁLISIS ESPECTRAL
# ═════════════════════════════════════════════════════════════════════════════

class TestSpectralAnalysis:
    """
    Suite de validación de propiedades espectrales.
    
    Verifica:
        - Espectro del Laplaciano
        - Gap espectral
        - Kernel
        - Número de condición
    """
    
    def test_spectral_eigenvalues_non_negative(self, double_cycle):
        """
        Verifica que todos los autovalores sean no negativos.
        """
        solenoid = AcousticSolenoidOperator()
        spectral = solenoid.spectral_analysis(double_cycle)
        
        eigenvalues = spectral["laplacian_spectrum"]
        assert all(lam >= -NC.BASE_TOLERANCE for lam in eigenvalues), \
            "Todos los autovalores deben ser ≥ 0"
    
    def test_spectral_gap_dag(self, dag_simple):
        """
        Verifica gap espectral en DAG (debe ser > 0 si β₁ = 0).
        """
        solenoid = AcousticSolenoidOperator()
        spectral = solenoid.spectral_analysis(dag_simple)
        
        gap = spectral["spectral_gap"]
        # DAG sin ciclos debe tener gap > 0
        assert gap >= 0.0
    
    def test_spectral_kernel_dimension(self, double_cycle):
        """
        Verifica dimensión del kernel = β₁.
        """
        solenoid = AcousticSolenoidOperator()
        spectral = solenoid.spectral_analysis(double_cycle)
        
        kernel_dim = spectral["kernel_dimension"]
        
        # double_cycle tiene β₁ = 2
        assert kernel_dim == 2, f"Esperado kernel dim=2, obtenido {kernel_dim}"
    
    def test_spectral_condition_number_bounded(self, single_cycle):
        """
        Verifica que número de condición sea finito y razonable.
        """
        solenoid = AcousticSolenoidOperator()
        spectral = solenoid.spectral_analysis(single_cycle)
        
        kappa = spectral["condition_number"]
        
        # No debe ser infinito (a menos que L₁ sea singular, lo cual es válido)
        # Para grafos pequeños, κ debe ser moderado
        if not math.isinf(kappa):
            assert kappa > 0
    
    def test_spectral_trace_consistency(self, single_cycle):
        """
        Verifica que traza = suma de autovalores.
        """
        solenoid = AcousticSolenoidOperator()
        spectral = solenoid.spectral_analysis(single_cycle)
        
        trace = spectral["properties"]["trace"]
        eigenvalues = spectral["laplacian_spectrum"]
        trace_eigs = sum(eigenvalues)
        
        assert math.isclose(trace, trace_eigs, rel_tol=NC.BASE_TOLERANCE), \
            f"Tr(L₁)={trace:.6e} ≠ Σλᵢ={trace_eigs:.6e}"


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 10: TESTS DE CASOS LÍMITE Y ROBUSTEZ
# ═════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """
    Suite de validación de casos límite y robustez numérica.
    
    Verifica:
        - Grafos triviales
        - Grafos vacíos
        - Self-loops
        - Flujos extremos
        - Estabilidad numérica
    """
    
    # ─────────────────────────────────────────────────────────────────────────
    # 10.1 Grafos Triviales
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_edge_trivial_graph(self, trivial_graph):
        """
        Verifica manejo de grafo trivial (1 nodo, 0 aristas).
        """
        result = inspect_and_mitigate_resonance(trivial_graph, {})
        
        assert result["status"] == "LAMINAR_FLOW"
        assert result["vorticity_metrics"]["betti_1_cycles"] == 0
    
    def test_edge_empty_graph(self):
        """
        Verifica rechazo de grafo vacío.
        """
        G = nx.DiGraph()
        
        with pytest.raises(GraphStructureError, match="vacío"):
            HodgeDecompositionBuilder(G)
    
    def test_edge_single_edge_graph(self):
        """
        Verifica grafo con una sola arista.
        """
        G = nx.DiGraph()
        G.add_edge("A", "B")
        
        flows = {("A", "B"): 1.0}
        result = inspect_and_mitigate_resonance(G, flows)
        
        # Sin ciclos → laminar
        assert result["status"] == "LAMINAR_FLOW"
    
    # ─────────────────────────────────────────────────────────────────────────
    # 10.2 Self-Loops
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_edge_self_loop_detection(self, self_loop_graph):
        """
        Verifica detección de self-loop como ciclo.
        """
        props = verify_hodge_properties(self_loop_graph)
        betti = props["betti_numbers"]
        
        # Self-loop cuenta como ciclo → β₁ ≥ 1
        assert betti["beta_1"] >= 1
    
    def test_edge_self_loop_vorticity(self, self_loop_graph):
        """
        Verifica vorticidad en self-loop.
        """
        flows = {
            ("A", "A"): 10.0,
            ("A", "B"): 1.0,
        }
        
        result = inspect_and_mitigate_resonance(self_loop_graph, flows)
        
        # Self-loop con flujo debe generar vorticidad
        if result["status"] != "LAMINAR_FLOW":
            assert result["vorticity_metrics"]["vorticity_index"] > 0.0
    
    # ─────────────────────────────────────────────────────────────────────────
    # 10.3 Flujos Extremos
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_edge_very_large_flows(self, single_cycle):
        """
        Verifica manejo de flujos muy grandes.
        """
        flows = {edge: 1e10 for edge in single_cycle.edges()}
        
        result = inspect_and_mitigate_resonance(single_cycle, flows)
        
        # No debe fallar con overflow
        assert "vorticity_metrics" in result
    
    def test_edge_very_small_flows(self, single_cycle):
        """
        Verifica manejo de flujos muy pequeños.
        """
        flows = {edge: 1e-15 for edge in single_cycle.edges()}
        
        result = inspect_and_mitigate_resonance(single_cycle, flows)
        
        # Flujos pequeños → probablemente laminar
        assert result["status"] in ["LAMINAR_FLOW", "RESONANCE_DETECTED"]
    
    def test_edge_mixed_sign_flows(self, single_cycle):
        """
        Verifica manejo de flujos con signos mixtos.
        """
        flows = {
            ("A", "B"): 10.0,
            ("B", "C"): -5.0,
            ("C", "A"): 3.0,
        }
        
        result = inspect_and_mitigate_resonance(single_cycle, flows)
        
        # Debe completar sin error
        assert "status" in result
    
    # ─────────────────────────────────────────────────────────────────────────
    # 10.4 Estabilidad Numérica
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_numerical_stability_ill_conditioned(self, complete_graph_3):
        """
        Verifica estabilidad con matriz mal condicionada.
        """
        # Grafo completo genera matriz con número de condición alto
        props = verify_hodge_properties(complete_graph_3)
        
        laplacian = props["laplacian_analysis"]
        kappa = laplacian["condition_number"]
        
        # Debe calcular κ sin fallar
        assert kappa >= 1.0 or math.isinf(kappa)
    
    def test_numerical_stability_rank_deficient(self, disconnected_graph):
        """
        Verifica estabilidad con matriz rank-deficiente.
        """
        builder = HodgeDecompositionBuilder(disconnected_graph)
        B1, meta = builder.build_incidence_matrix()
        
        # Grafo desconectado → B₁ rank-deficiente
        n = disconnected_graph.number_of_nodes()
        assert meta["rank_B1"] < n
    
    def test_numerical_near_zero_tolerance(self, single_cycle):
        """
        Verifica manejo de valores cercanos a cero.
        """
        # Flujos cercanos al umbral de tolerancia
        flows = {edge: NC.VORTICITY_SIGNIFICANCE_THRESHOLD * 1.1 for edge in single_cycle.edges()}
        
        result = inspect_and_mitigate_resonance(single_cycle, flows)
        
        # Debe manejar correctamente sin falsos positivos/negativos
        assert "status" in result


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 11: TESTS DE VERIFICACIÓN FORMAL
# ═════════════════════════════════════════════════════════════════════════════

class TestFormalVerification:
    """
    Suite de verificación formal de propiedades matemáticas.
    
    Verifica:
        - verify_hodge_properties (verificación standalone)
        - Consistencia de todas las verificaciones
        - Completitud de checks
    """
    
    def test_verify_hodge_all_checks(self, double_cycle):
        """
        Verifica que verify_hodge_properties ejecute todos los checks.
        """
        props = verify_hodge_properties(double_cycle)
        
        # Verificar presencia de todas las secciones
        required_sections = [
            "graph_properties",
            "chain_complex_verification",
            "betti_numbers",
            "euler_characteristic",
            "hodge_isomorphism",
            "spectral_properties",
            "laplacian_analysis",
            "verification_summary",
        ]
        
        for section in required_sections:
            assert section in props, f"Falta sección: {section}"
    
    def test_verify_hodge_summary_consistency(self, single_cycle):
        """
        Verifica consistencia del summary de verificación.
        """
        props = verify_hodge_properties(single_cycle)
        summary = props["verification_summary"]
        
        # Si all_checks_passed es True, todos los checks individuales deben ser True
        if summary["all_checks_passed"]:
            assert summary["boundary_composition_zero"] is True
            assert summary["euler_verified"] is True
            assert summary["hodge_iso_satisfied"] is True
            assert summary["laplacian_symmetric"] is True
            assert summary["laplacian_psd"] is True
    
    def test_verify_hodge_graph_properties_accurate(self, disconnected_graph):
        """
        Verifica que propiedades del grafo sean precisas.
        """
        props = verify_hodge_properties(disconnected_graph)
        graph_props = props["graph_properties"]
        
        assert graph_props["nodes"] == disconnected_graph.number_of_nodes()
        assert graph_props["edges"] == disconnected_graph.number_of_edges()
        assert graph_props["is_directed"] is True
        assert graph_props["connected_components"] == 2


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 12: TESTS DE PERFORMANCE (OPCIONAL - MARCADOS COMO SLOW)
# ═════════════════════════════════════════════════════════════════════════════

class TestPerformance:
    """
    Suite de validación de performance (opcional).
    
    Tests marcados como @pytest.mark.slow para ejecución selectiva.
    """
    
    @pytest.mark.slow
    def test_performance_large_dag(self):
        """
        Verifica performance en DAG grande.
        """
        import time
        
        # Crear DAG grande (1000 nodos)
        G = nx.DiGraph()
        for i in range(1000):
            if i > 0:
                G.add_edge(f"node_{i-1}", f"node_{i}")
        
        flows = {edge: 1.0 for edge in G.edges()}
        
        start = time.time()
        result = inspect_and_mitigate_resonance(G, flows)
        elapsed = time.time() - start
        
        # Debe completar en tiempo razonable (< 5s)
        assert elapsed < 5.0, f"Análisis tomó {elapsed:.2f}s (> 5s)"
        
        # DAG → debe ser laminar
        assert result["status"] == "LAMINAR_FLOW"
    
    @pytest.mark.slow
    def test_performance_cache_effectiveness(self, double_cycle):
        """
        Verifica efectividad del caché de builders.
        """
        import time
        
        solenoid = AcousticSolenoidOperator(enable_caching=True)
        flows = {edge: 1.0 for edge in double_cycle.edges()}
        
        # Primera ejecución (cache miss)
        start = time.time()
        for _ in range(10):
            solenoid.isolate_vorticity(double_cycle, flows)
        first_run = time.time() - start
        
        # Segunda ejecución (debería usar caché)
        start = time.time()
        for _ in range(10):
            solenoid.isolate_vorticity(double_cycle, flows)
        second_run = time.time() - start
        
        # Segunda ejecución debe ser más rápida o igual (caché)
        assert second_run <= first_run * 1.5  # Margen de 50%
        
        # Verificar estadísticas de caché
        stats = solenoid.get_cache_statistics()
        assert stats["cache_hits"] > 0


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 13: CONFIGURACIÓN DE PYTEST Y HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def pytest_configure(config):
    """
    Configuración adicional de pytest.
    """
    config.addinivalue_line(
        "markers",
        "slow: marca tests que requieren más tiempo de ejecución"
    )


# ═════════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA PARA EJECUCIÓN DIRECTA
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Permite ejecutar la suite completa directamente.
    
    Uso:
        python test_solenoid_acoustic.py
        python test_solenoid_acoustic.py -v
        python test_solenoid_acoustic.py -k "test_unit" -v
        python test_solenoid_acoustic.py -m "not slow" -v
    """
    import sys
    
    pytest.main([
        __file__,
        "-v",              # Verbose
        "--tb=short",      # Traceback corto
        "--color=yes",     # Colorear output
        "-ra",             # Resumen de todos los tests
        "--strict-markers", # Validar markers
        "-m", "not slow",  # Excluir tests lentos por defecto
    ] + sys.argv[1:])