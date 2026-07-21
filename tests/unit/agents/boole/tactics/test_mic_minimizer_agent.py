# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo de Pruebas: MIC Minimizer Agent Test Suite                            ║
║ Ruta: tests/unit/agents/boole/tactics/test_mic_minimizer_agent.py            ║
║ Versión: 3.0.0-Boolean-Rigorous-Test-Suite                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

ARQUITECTURA DE TESTING RIGUROSO:
────────────────────────────────────────────────────────────────────────────────
Esta suite de pruebas valida exhaustivamente la corrección algebraica y numérica
del MICMinimizerAgent mediante:

1. PRUEBAS UNITARIAS ATÓMICAS:
   - Validación de matrices sobre GF(2)
   - Eliminación de Gauss-Jordan sobre campos finitos
   - Forma escalonada reducida (RREF)
   - Cómputo de kernel y espacios nulos
   - Entropías de Shannon, Rényi y familia completa

2. PRUEBAS DE INTEGRACIÓN POR FASE:
   - Fase 1: Bases de Gröbner y teoría de códigos
   - Fase 2: Ortogonalidad y matriz de Gram
   - Fase 3: Equivalencia homotópica ROBDD

3. PRUEBAS DE COMPOSICIÓN FUNTORIAL:
   - Pipeline completo Φ₃ ∘ Φ₂ ∘ Φ₁
   - Propagación de certificados entre fases
   - Validación de invariantes categóricos

4. PRUEBAS DE CASOS EXTREMOS:
   - Matrices de rango completo y deficiente
   - Códigos lineales degenerados
   - Distribuciones uniformes y concentradas
   - Matrices dispersas y densas

5. PRUEBAS DE ROBUSTEZ NUMÉRICA:
   - Perturbaciones de ortogonalidad
   - Matrices mal condicionadas
   - Distribuciones con soporte pequeño

6. PRUEBAS DE MANEJO DE EXCEPCIONES:
   - Degeneración de Gröbner
   - Violaciones de ortogonalidad
   - Pérdida de entropía

COBERTURA OBJETIVO:
────────────────────────────────────────────────────────────────────────────────
- Cobertura de líneas: ≥ 95%
- Cobertura de ramas: ≥ 90%
- Cobertura de condiciones: ≥ 85%
- Cobertura de mutaciones: ≥ 80%

HERRAMIENTAS UTILIZADAS:
────────────────────────────────────────────────────────────────────────────────
- pytest: Framework de testing principal
- hypothesis: Property-based testing
- numpy.testing: Aserciones numéricas con tolerancia
- pytest-cov: Análisis de cobertura
- pytest-benchmark: Benchmarking de rendimiento
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable, List, Set, Tuple

import numpy as np
import pytest
import scipy.linalg as la
from hypothesis import given, settings, strategies as st
from numpy.testing import assert_allclose, assert_array_equal

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE RUTAS PARA IMPORTACIÓN
# ═══════════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTACIONES DEL MÓDULO BAJO PRUEBA
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from app.agents.boole.tactics.mic_minimizer_agent import (
        BooleanInputValidationError,
        BooleanReductionPhase,
        EntropyAnalysis,
        GF2MatrixProperties,
        GrobnerAuditData,
        GrobnerDegeneracyError,
        LinearCodeProperties,
        LinearDependencyError,
        MICMinimizerAgent,
        MICMinimizerAgentError,
        MinimizerGovernanceState,
        NonInterferenceViolationError,
        OrthogonalityAnalysis,
        OrthogonalityMetric,
        Phase1_GrobnerBasisAuditor,
        Phase2_UnsatCoreCertifier,
        Phase3_ROBDDIsomorphismValidator,
        ProbabilityDistributionError,
        ROBDDHomotopyError,
        ROBDDIsomorphismData,
        UnsatCoreCertifierData,
        _ENTROPY_TOLERANCE,
        _KL_DIVERGENCE_TOLERANCE,
        _MACHINE_EPSILON,
        _MAX_BOOLEAN_VARIABLES,
        _MIN_ENTROPY_TOLERANCE,
        _ORTHOGONALITY_TOLERANCE,
        _PROBABILITY_TOLERANCE,
    )
except ImportError as exc:
    pytest.skip(
        f"No se pudo importar el módulo mic_minimizer_agent: {exc}",
        allow_module_level=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES DE TESTING
# ═══════════════════════════════════════════════════════════════════════════════
NUMERICAL_TOLERANCE = 1e-10
STRICT_TOLERANCE = 1e-12
RELAXED_TOLERANCE = 1e-8

# Semillas para reproducibilidad
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES: GENERADORES DE MATRICES SOBRE GF(2)
# ═══════════════════════════════════════════════════════════════════════════════
@pytest.fixture
def identity_matrix_gf2() -> np.ndarray:
    r"""
    Genera matriz identidad 3×3 sobre GF(2).
    
    Propiedades:
        - Rango: 3
        - Nulidad: 0
        - Base canónica
    """
    return np.eye(3, dtype=np.uint8)


@pytest.fixture
def full_rank_matrix_gf2() -> np.ndarray:
    r"""
    Genera matriz de rango completo sobre GF(2).
    
    Matriz 3×4 con rango 3.
    """
    return np.array([
        [1, 0, 1, 0],
        [0, 1, 1, 0],
        [1, 1, 0, 1],
    ], dtype=np.uint8)


@pytest.fixture
def rank_deficient_matrix_gf2() -> np.ndarray:
    r"""
    Genera matriz de rango deficiente sobre GF(2).
    
    Matriz 3×3 con rango 2 (fila 3 = fila 1 + fila 2 mod 2).
    """
    return np.array([
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 0],  # XOR de las dos primeras
    ], dtype=np.uint8)


@pytest.fixture
def hamming_code_matrix() -> np.ndarray:
    r"""
    Genera matriz generadora del código de Hamming [7,4,3].
    
    Propiedades:
        - Código perfecto
        - Distancia mínima: 3
        - Corrige 1 error
    """
    return np.array([
        [1, 1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0],
        [1, 1, 1, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 0, 1],
    ], dtype=np.uint8)


@pytest.fixture
def random_gf2_matrix() -> Callable[[int, int], np.ndarray]:
    r"""
    Factory para generar matrices aleatorias sobre GF(2).
    
    Returns:
        Función que genera matriz m×n sobre GF(2)
    """
    def _generate(rows: int, cols: int) -> np.ndarray:
        return np.random.randint(0, 2, size=(rows, cols), dtype=np.uint8)
    
    return _generate


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES: GENERADORES DE MATRICES DE PROYECCIÓN
# ═══════════════════════════════════════════════════════════════════════════════
@pytest.fixture
def orthonormal_projection_2d() -> np.ndarray:
    r"""
    Genera matriz de proyección ortogonal 2×3.
    
    Filas ortonormales: ‖pᵢ‖ = 1, ⟨pᵢ, pⱼ⟩ = 0 para i ≠ j.
    """
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)


@pytest.fixture
def orthonormal_projection_3d() -> np.ndarray:
    r"""
    Genera matriz de proyección ortogonal 3×3.
    
    Matriz identidad (caso trivial de ortogonalidad perfecta).
    """
    return np.eye(3, dtype=np.float64)


@pytest.fixture
def non_orthogonal_projection() -> np.ndarray:
    r"""
    Genera matriz de proyección NO ortogonal.
    
    Filas no ortogonales: ⟨p₁, p₂⟩ ≠ 0.
    """
    return np.array([
        [1.0, 0.0, 0.0],
        [0.5, 0.866, 0.0],  # Ángulo de 30° con la primera
    ], dtype=np.float64)


@pytest.fixture
def random_projection_matrix() -> Callable[[int, int], np.ndarray]:
    r"""
    Factory para generar matrices de proyección aleatorias.
    
    Returns:
        Función que genera matriz m×n ortogonalizada por Gram-Schmidt
    """
    def _generate(rows: int, cols: int, orthogonal: bool = True) -> np.ndarray:
        M = np.random.randn(rows, cols)
        
        if orthogonal and rows <= cols:
            # Ortogonalización de Gram-Schmidt
            Q, _ = np.linalg.qr(M.T)
            return Q.T[:rows, :]
        else:
            # Normalizar filas
            norms = np.linalg.norm(M, axis=1, keepdims=True)
            return M / np.where(norms > 1e-10, norms, 1.0)
    
    return _generate


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES: GENERADORES DE DISTRIBUCIONES PROBABILÍSTICAS
# ═══════════════════════════════════════════════════════════════════════════════
@pytest.fixture
def uniform_distribution() -> np.ndarray:
    r"""
    Genera distribución uniforme de 8 elementos.
    
    p(x) = 1/8 ∀x
    
    Propiedades:
        - Entropía máxima: H(X) = log₂(8) = 3 bits
        - Uniformidad: 1.0
    """
    return np.ones(8, dtype=np.float64) / 8.0


@pytest.fixture
def concentrated_distribution() -> np.ndarray:
    r"""
    Genera distribución concentrada.
    
    p(x₀) = 0.9, p(x₁) = 0.1
    
    Propiedades:
        - Entropía baja
        - Min-entropía alta
    """
    return np.array([0.9, 0.1], dtype=np.float64)


@pytest.fixture
def random_distribution() -> Callable[[int], np.ndarray]:
    r"""
    Factory para generar distribuciones aleatorias.
    
    Usa distribución de Dirichlet para generar probabilidades válidas.
    
    Returns:
        Función que genera distribución de n elementos
    """
    def _generate(size: int, alpha: float = 1.0) -> np.ndarray:
        # Distribución de Dirichlet con parámetro α
        p = np.random.dirichlet(alpha=np.ones(size) * alpha)
        return p / np.sum(p)  # Normalización extra por seguridad
    
    return _generate


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES: INSTANCIAS DE AGENTES
# ═══════════════════════════════════════════════════════════════════════════════
@pytest.fixture
def phase1_auditor() -> Phase1_GrobnerBasisAuditor:
    """Instancia del auditor de Fase 1."""
    return Phase1_GrobnerBasisAuditor()


@pytest.fixture
def phase2_certifier() -> Phase2_UnsatCoreCertifier:
    """Instancia del certificador de Fase 2."""
    return Phase2_UnsatCoreCertifier()


@pytest.fixture
def phase3_validator() -> Phase3_ROBDDIsomorphismValidator:
    """Instancia del validador de Fase 3."""
    return Phase3_ROBDDIsomorphismValidator()


@pytest.fixture
def mic_agent() -> MICMinimizerAgent:
    """Instancia del agente completo."""
    return MICMinimizerAgent()


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   SUITE 1: PRUEBAS DE VALIDACIÓN DE MATRICES GF(2)                          ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class TestGF2MatrixValidation:
    """Pruebas de validación de matrices sobre GF(2)."""
    
    def test_identity_matrix_is_valid(
        self,
        phase1_auditor: Phase1_GrobnerBasisAuditor,
        identity_matrix_gf2: np.ndarray,
    ) -> None:
        """Verifica que la matriz identidad sea válida sobre GF(2)."""
        props = phase1_auditor._compute_gf2_matrix_properties(
            identity_matrix_gf2,
            "identity",
        )
        
        assert props.rows == 3
        assert props.cols == 3
        assert props.rank_gf2 == 3
        assert props.nullity_gf2 == 0
        assert props.has_full_rank
        assert props.kernel_dimension == 0
    
    def test_full_rank_matrix_properties(
        self,
        phase1_auditor: Phase1_GrobnerBasisAuditor,
        full_rank_matrix_gf2: np.ndarray,
    ) -> None:
        """Verifica propiedades de matriz de rango completo."""
        props = phase1_auditor._compute_gf2_matrix_properties(
            full_rank_matrix_gf2,
            "full_rank",
        )
        
        assert props.rows == 3
        assert props.cols == 4
        assert props.rank_gf2 == 3
        assert props.nullity_gf2 == 1
        assert props.has_full_rank
        
        # Verificar teorema del rango-nulidad
        assert props.rank_gf2 + props.nullity_gf2 == props.cols
    
    def test_rank_deficient_matrix_detected(
        self,
        phase1_auditor: Phase1_GrobnerBasisAuditor,
        rank_deficient_matrix_gf2: np.ndarray,
    ) -> None:
        """Verifica detección de rango deficiente."""
        props = phase1_auditor._compute_gf2_matrix_properties(
            rank_deficient_matrix_gf2,
            "rank_deficient",
        )
        
        assert props.rows == 3
        assert props.cols == 3
        assert props.rank_gf2 == 2  # Rango deficiente
        assert props.nullity_gf2 == 1
        assert not props.has_full_rank
    
    def test_non_binary_matrix_rejected(
        self,
        phase1_auditor: Phase1_GrobnerBasisAuditor,
    ) -> None:
        """Verifica que matrices no binarias sean rechazadas."""
        bad_matrix = np.array([[1, 2], [0, 1]], dtype=np.uint8)
        
        with pytest.raises(BooleanInputValidationError, match="fuera de GF\\(2\\)"):
            phase1_auditor._as_finite_gf2_matrix("bad_matrix", bad_matrix)
    
    def test_empty_matrix_rejected(
        self,
        phase1_auditor: Phase1_GrobnerBasisAuditor,
    ) -> None:
        """Verifica que matriz vacía sea rechazada."""
        empty_matrix = np.array([], dtype=np.uint8).reshape(0, 0)
        
        with pytest.raises(BooleanInputValidationError, match="vacía"):
            phase1_auditor._audit_grobner_independence(empty_matrix)
    
    def test_sparsity_computation(
        self,
        phase1_auditor: Phase1_GrobnerBasisAuditor,
    ) -> None:
        """Verifica cómputo de sparsity."""
        sparse_matrix = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0],
        ], dtype=np.uint8)
        
        props = phase1_auditor._compute_gf2_matrix_properties(
            sparse_matrix,
            "sparse",
        )
        
        # Sparsity = número de ceros / total
        expected_sparsity = 10.0 / 12.0
        assert_allclose(props.sparsity, expected_sparsity, atol=NUMERICAL_TOLERANCE)
    
    def test_hamming_weight_computation(
        self,
        phase1_auditor: Phase1_GrobnerBasisAuditor,
    ) -> None:
        """Verifica cómputo de peso de Hamming."""
        matrix = np.array([
            [1, 1, 0],
            [0, 1, 1],
        ], dtype=np.uint8)
        
        props = phase1_auditor._compute_gf2_matrix_properties(matrix, "test")
        
        # Peso de Hamming = número de unos
        expected_weight = 4
        assert props.hamming_weight == expected_weight


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   SUITE 2: PRUEBAS DE ELIMINACIÓN DE GAUSS-JORDAN SOBRE GF(2)               ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class TestGaussJordanGF2:
    """Pruebas de eliminación de Gauss-Jordan sobre GF(2)."""
    
    def test_rref_identity_matrix(
        self,
        phase1_auditor: Phase1_GrobnerBasisAuditor,
        identity_matrix_gf2: np.ndarray,
    ) -> None:
        """Verifica que RREF de identidad sea identidad."""
        rref, pivots = phase1_auditor._gf2_rref(identity_matrix_gf2)
        
        assert_array_equal(rref, identity_matrix_gf2)
        assert pivots == (0, 1, 2)
    
    def test_rref_full_rank_matrix(
        self,
        phase1_auditor: Phase1_GrobnerBasisAuditor,
        full_rank_matrix_gf2: np.ndarray,
    ) -> None:
        """Verifica RREF de matriz de rango completo."""
        rref, pivots = phase1_auditor._gf2_rref(full_rank_matrix_gf2)
        
        # Debe tener 3 pivotes (rango 3)
        assert len(pivots) == 3
        
        # Verificar forma escalonada
        for i, pivot_col in enumerate(pivots):
            # Pivote debe ser 1
            assert rref[i, pivot_col] == 1
            
            # Resto de la columna debe ser 0
            for j in range(rref.shape[0]):
                if j != i:
                    assert rref[j, pivot_col] == 0
    
    def test_rank_computation_via_rref(
        self,
        phase1_auditor: Phase1_GrobnerBasisAuditor,
        rank_deficient_matrix_gf2: np.ndarray,
    ) -> None:
        """Verifica cómputo de rango via RREF."""
        rank, pivots = phase1_auditor._gf2_rank(rank_deficient_matrix_gf2)
        
        assert rank == 2
        assert len(pivots) == 2
    
    def test_gf2_arithmetic_correctness(
        self,
        phase1_auditor: Phase1_GrobnerBasisAuditor,
    ) -> None:
        """Verifica aritmética módulo 2."""
        matrix = np.array([
            [1, 1],
            [1, 1],
        ], dtype=np.uint8)
        
        rref, pivots = phase1_auditor._gf2_rref(matrix)
        
        # Fila 2 = fila 1, por lo que se elimina
        expected_rref = np.array([
            [1, 1],
            [0, 0],
        ], dtype=np.uint8)
        
        assert_array_equal(rref, expected_rref)
        assert len(pivots) == 1


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   SUITE 3: PRUEBAS DE KERNEL Y ESPACIOS NULOS                               ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class TestKernelComputation:
    """Pruebas de cómputo de kernel sobre GF(2)."""
    
    def test_kernel_of_identity_is_trivial(
        self,
        phase1_auditor: Phase1_GrobnerBasisAuditor,
        identity_matrix_gf2: np.ndarray,
    ) -> None:
        """Verifica que kernel de identidad sea trivial."""
        kernel = phase1_auditor._compute_kernel_gf2(identity_matrix_gf2)
        
        # Kernel trivial: ninguna columna
        assert kernel.shape == (3, 0)
    
    def test_kernel_dimension_matches_nullity(
        self,
        phase1_auditor: Phase1_GrobnerBasisAuditor,
        full_rank_matrix_gf2: np.ndarray,
    ) -> None:
        """Verifica que dimensión del kernel = nulidad."""
        kernel = phase1_auditor._compute_kernel_gf2(full_rank_matrix_gf2)
        
        # Matriz 3×4 de rango 3 → nulidad = 1
        assert kernel.shape[1] == 1
        
        # Verificar que vectores del kernel son soluciones: Mv = 0
        for i in range(kernel.shape[1]):
            v = kernel[:, i]
            product = full_rank_matrix_gf2 @ v
            product_mod2 = np.mod(product, 2)
            
            expected = np.zeros(full_rank_matrix_gf2.shape[0], dtype=np.uint8)
            assert_array_equal(product_mod2, expected)
    
    def test_kernel_vectors_are_independent(
        self,
        phase1_auditor: Phase1_GrobnerBasisAuditor,
    ) -> None:
        """Verifica que vectores del kernel sean linealmente independientes."""
        # Matriz con nulidad 2
        matrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 1, 0],
        ], dtype=np.uint8)
        
        kernel = phase1_auditor._compute_kernel_gf2(matrix)
        
        # Nulidad = 4 - 2 = 2
        assert kernel.shape[1] == 2
        
        # Verificar independencia: rango del kernel = 2
        if kernel.shape[1] > 0:
            rank_kernel, _ = phase1_auditor._gf2_rank(kernel)
            assert rank_kernel == kernel.shape[1]


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   SUITE 4: PRUEBAS DE DETECCIÓN DE DEPENDENCIAS LINEALES                    ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class TestLinearDependencyDetection:
    """Pruebas de detección de dependencias lineales."""
    
    def test_no_dependencies_in_independent_matrix(
        self,
        phase1_auditor: Phase1_GrobnerBasisAuditor,
        full_rank_matrix_gf2: np.ndarray,
    ) -> None:
        """Verifica que matriz independiente no tenga dependencias."""
        dependencies = phase1_auditor._detect_linear_dependencies(
            full_rank_matrix_gf2
        )
        
        # No debe haber dependencias
        assert len(dependencies) == 0
    
    def test_dependency_detected_in_rank_deficient_matrix(
        self,
        phase1_auditor: Phase1_GrobnerBasisAuditor,
        rank_deficient_matrix_gf2: np.ndarray,
    ) -> None:
        """Verifica detección de dependencia en matriz de rango deficiente."""
        dependencies = phase1_auditor._detect_linear_dependencies(
            rank_deficient_matrix_gf2
        )
        
        # Debe haber al menos una dependencia
        assert len(dependencies) > 0
        
        # La dependencia debe involucrar a las 3 filas
        for dep in dependencies:
            assert len(dep) >= 2
    
    def test_explicit_dependency_verification(
        self,
        phase1_auditor: Phase1_GrobnerBasisAuditor,
    ) -> None:
        """Verifica dependencia explícita conocida."""
        # Fila 3 = fila 1 + fila 2 (mod 2)
        matrix = np.array([
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0],
        ], dtype=np.uint8)
        
        dependencies = phase1_auditor._detect_linear_dependencies(matrix)
        
        assert len(dependencies) > 0
        
        # Verificar que las 3 filas estén en alguna dependencia
        all_involved = set()
        for dep in dependencies:
            all_involved.update(dep)
        
        assert len(all_involved) >= 2


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   SUITE 5: PRUEBAS DE TEORÍA DE CÓDIGOS LINEALES                            ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class TestLinearCodeProperties:
    """Pruebas de propiedades de códigos lineales."""
    
    def test_hamming_code_properties(
        self,
        phase1_auditor: Phase1_GrobnerBasisAuditor,
        hamming_code_matrix: np.ndarray,
    ) -> None:
        """Verifica propiedades del código de Hamming [7,4,3]."""
        rank, _ = phase1_auditor._gf2_rank(hamming_code_matrix)
        
        code_props = phase1_auditor._compute_code_properties(
            hamming_code_matrix,
            rank,
        )
        
        assert code_props.length == 7
        assert code_props.dimension == 4
        assert code_props.minimum_distance >= 3  # Hamming tiene d=3
        
        # Verificar tasa del código
        expected_rate = 4.0 / 7.0
        assert_allclose(code_props.rate, expected_rate, atol=NUMERICAL_TOLERANCE)
    
    def test_singleton_bound(
        self,
        phase1_auditor: Phase1_GrobnerBasisAuditor,
    ) -> None:
        """Verifica cota de Singleton: d ≤ n - k + 1."""
        matrix = np.array([
            [1, 0, 1],
            [0, 1, 1],
        ], dtype=np.uint8)
        
        rank, _ = phase1_auditor._gf2_rank(matrix)
        code_props = phase1_auditor._compute_code_properties(matrix, rank)
        
        # d ≤ n - k + 1
        assert code_props.minimum_distance <= code_props.singleton_bound
    
    def test_code_rate_bounds(
        self,
        phase1_auditor: Phase1_GrobnerBasisAuditor,
        full_rank_matrix_gf2: np.ndarray,
    ) -> None:
        """Verifica que la tasa del código esté en [0, 1]."""
        rank, _ = phase1_auditor._gf2_rank(full_rank_matrix_gf2)
        code_props = phase1_auditor._compute_code_properties(
            full_rank_matrix_gf2,
            rank,
        )
        
        assert 0.0 <= code_props.rate <= 1.0
    
    def test_redundancy_calculation(
        self,
        phase1_auditor: Phase1_GrobnerBasisAuditor,
        hamming_code_matrix: np.ndarray,
    ) -> None:
        """Verifica cálculo de redundancia."""
        rank, _ = phase1_auditor._gf2_rank(hamming_code_matrix)
        code_props = phase1_auditor._compute_code_properties(
            hamming_code_matrix,
            rank,
        )
        
        # Redundancia = n - k
        expected_redundancy = 7 - 4
        assert code_props.redundancy == expected_redundancy


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   SUITE 6: PRUEBAS DE FASE 1 (BASES DE GRÖBNER)                             ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class TestPhase1_GrobnerAudit:
    """Pruebas de auditoría de bases de Gröbner."""
    
    def test_independent_basis_accepted(
        self,
        phase1_auditor: Phase1_GrobnerBasisAuditor,
        full_rank_matrix_gf2: np.ndarray,
    ) -> None:
        """Verifica que base independiente sea aceptada."""
        audit = phase1_auditor._audit_grobner_independence(
            full_rank_matrix_gf2
        )
        
        assert audit.is_minimally_independent
        assert audit.ideal_dimension == 3
        assert audit.nullity == 1
    
    def test_rank_deficient_basis_rejected(
        self,
        phase1_auditor: Phase1_GrobnerBasisAuditor,
        rank_deficient_matrix_gf2: np.ndarray,
    ) -> None:
        """Verifica que base de rango deficiente sea rechazada."""
        with pytest.raises(GrobnerDegeneracyError, match="Degeneración"):
            phase1_auditor._audit_grobner_independence(
                rank_deficient_matrix_gf2
            )
    
    def test_identity_basis_properties(
        self,
        phase1_auditor: Phase1_GrobnerBasisAuditor,
        identity_matrix_gf2: np.ndarray,
    ) -> None:
        """Verifica propiedades de base identidad."""
        audit = phase1_auditor._audit_grobner_independence(
            identity_matrix_gf2
        )
        
        assert audit.is_minimally_independent
        assert audit.ideal_dimension == 3
        assert audit.nullity == 0
        assert audit.matrix_properties.has_full_rank
    
    def test_rref_matrix_stored(
        self,
        phase1_auditor: Phase1_GrobnerBasisAuditor,
        full_rank_matrix_gf2: np.ndarray,
    ) -> None:
        """Verifica que matriz RREF se almacene en certificado."""
        audit = phase1_auditor._audit_grobner_independence(
            full_rank_matrix_gf2
        )
        
        assert audit.rref_matrix is not None
        assert audit.rref_matrix.shape == full_rank_matrix_gf2.shape
    
    def test_dependency_graph_analysis(
        self,
        phase1_auditor: Phase1_GrobnerBasisAuditor,
        full_rank_matrix_gf2: np.ndarray,
    ) -> None:
        """Verifica análisis de grafo de dependencias."""
        audit = phase1_auditor._audit_grobner_independence(
            full_rank_matrix_gf2
        )
        
        # Matriz independiente no debe tener aristas de dependencia
        assert audit.dependency_graph_edges == 0


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   SUITE 7: PRUEBAS DE ORTOGONALIDAD Y MATRIZ DE GRAM                        ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class TestOrthogonalityAnalysis:
    """Pruebas de análisis de ortogonalidad."""
    
    def test_gram_matrix_identity_for_orthonormal(
        self,
        phase2_certifier: Phase2_UnsatCoreCertifier,
        orthonormal_projection_3d: np.ndarray,
    ) -> None:
        """Verifica que G = I para matriz ortonormal."""
        gram = phase2_certifier._compute_gram_matrix(orthonormal_projection_3d)
        
        identity = np.eye(3, dtype=np.float64)
        assert_allclose(gram, identity, atol=NUMERICAL_TOLERANCE)
    
    def test_orthogonality_analysis_perfect(
        self,
        phase2_certifier: Phase2_UnsatCoreCertifier,
        orthonormal_projection_3d: np.ndarray,
    ) -> None:
        """Verifica análisis de ortogonalidad perfecta."""
        gram = phase2_certifier._compute_gram_matrix(orthonormal_projection_3d)
        analysis = phase2_certifier._analyze_orthogonality(gram)
        
        # Desviaciones deben ser ~ 0
        assert analysis.off_diagonal_norm < NUMERICAL_TOLERANCE
        assert analysis.diagonal_deviation_norm < NUMERICAL_TOLERANCE
        assert analysis.frobenius_deviation < NUMERICAL_TOLERANCE
    
    def test_non_orthogonal_detected(
        self,
        phase2_certifier: Phase2_UnsatCoreCertifier,
        non_orthogonal_projection: np.ndarray,
    ) -> None:
        """Verifica detección de no ortogonalidad."""
        gram = phase2_certifier._compute_gram_matrix(non_orthogonal_projection)
        analysis = phase2_certifier._analyze_orthogonality(gram)
        
        # Debe haber desviación significativa fuera de diagonal
        assert analysis.off_diagonal_norm > NUMERICAL_TOLERANCE
    
    def test_conflict_pairs_detection(
        self,
        phase2_certifier: Phase2_UnsatCoreCertifier,
        non_orthogonal_projection: np.ndarray,
    ) -> None:
        """Verifica detección de pares en conflicto."""
        gram = phase2_certifier._compute_gram_matrix(non_orthogonal_projection)
        
        conflict_pairs = phase2_certifier._detect_conflict_pairs(
            gram,
            _ORTHOGONALITY_TOLERANCE,
        )
        
        # Debe detectar al menos un par en conflicto
        assert len(conflict_pairs) > 0
        
        # Los pares deben ser tuplas (i, j) con i < j
        for i, j in conflict_pairs:
            assert i < j
    
    def test_spectral_analysis(
        self,
        phase2_certifier: Phase2_UnsatCoreCertifier,
        orthonormal_projection_3d: np.ndarray,
    ) -> None:
        """Verifica análisis espectral de matriz de Gram."""
        gram = phase2_certifier._compute_gram_matrix(orthonormal_projection_3d)
        
        eigenvalues, spectral_radius, spectral_norm = (
            phase2_certifier._compute_spectral_analysis(gram)
        )
        
        # Para matriz identidad, todos los autovalores deben ser 1
        assert_allclose(eigenvalues, np.ones(3), atol=NUMERICAL_TOLERANCE)
        
        # Radio espectral y norma espectral deben ser 1
        assert_allclose(spectral_radius, 1.0, atol=NUMERICAL_TOLERANCE)
        assert_allclose(spectral_norm, 1.0, atol=NUMERICAL_TOLERANCE)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   SUITE 8: PRUEBAS DE FASE 2 (NO-INTERFERENCIA)                             ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class TestPhase2_UnsatCoreCertification:
    """Pruebas de certificación de no-interferencia."""
    
    def test_orthogonal_projection_certified(
        self,
        phase2_certifier: Phase2_UnsatCoreCertifier,
        orthonormal_projection_3d: np.ndarray,
    ) -> None:
        """Verifica que proyección ortogonal sea certificada."""
        audit = phase2_certifier._certify_non_interference_unsat(
            orthonormal_projection_3d
        )
        
        assert audit.is_strictly_orthogonal
        assert audit.conflict_edges == 0
    
    def test_non_orthogonal_projection_rejected(
        self,
        phase2_certifier: Phase2_UnsatCoreCertifier,
        non_orthogonal_projection: np.ndarray,
    ) -> None:
        """Verifica que proyección no ortogonal sea rechazada."""
        with pytest.raises(NonInterferenceViolationError, match="Zero Side-Effects"):
            phase2_certifier._certify_non_interference_unsat(
                non_orthogonal_projection
            )
    
    def test_projection_rank_computed(
        self,
        phase2_certifier: Phase2_UnsatCoreCertifier,
        orthonormal_projection_2d: np.ndarray,
    ) -> None:
        """Verifica cómputo de rango de proyección."""
        audit = phase2_certifier._certify_non_interference_unsat(
            orthonormal_projection_2d
        )
        
        # Proyección 2×3 de rango 2
        assert audit.projection_rank == 2
    
    def test_idempotence_validation(
        self,
        phase2_certifier: Phase2_UnsatCoreCertifier,
    ) -> None:
        """Verifica validación de idempotencia."""
        # Proyección idempotente: P² = P
        P = np.array([
            [1.0, 0.0],
            [0.0, 0.0],
        ], dtype=np.float64)
        
        rank, is_idempotent = phase2_certifier._validate_projection_properties(P)
        
        assert is_idempotent
    
    def test_certificate_propagation_from_phase1(
        self,
        phase1_auditor: Phase1_GrobnerBasisAuditor,
        phase2_certifier: Phase2_UnsatCoreCertifier,
        full_rank_matrix_gf2: np.ndarray,
        orthonormal_projection_2d: np.ndarray,
    ) -> None:
        """Verifica propagación de certificado de Fase 1 a Fase 2."""
        grobner_audit = phase1_auditor._audit_grobner_independence(
            full_rank_matrix_gf2
        )
        
        # Fase 2 debe aceptar certificado de Fase 1
        audit = phase2_certifier._certify_non_interference_unsat(
            orthonormal_projection_2d,
            grobner_audit=grobner_audit,
        )
        
        assert audit.is_strictly_orthogonal


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   SUITE 9: PRUEBAS DE ENTROPÍAS                                             ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class TestEntropyComputations:
    """Pruebas de cómputo de entropías."""
    
    def test_shannon_entropy_uniform(
        self,
        phase3_validator: Phase3_ROBDDIsomorphismValidator,
        uniform_distribution: np.ndarray,
    ) -> None:
        """Verifica entropía de Shannon para distribución uniforme."""
        entropy = phase3_validator._shannon_entropy_bits(uniform_distribution)
        
        # H(X) = log₂(8) = 3 bits para distribución uniforme de 8 elementos
        expected_entropy = math.log2(8.0)
        assert_allclose(entropy, expected_entropy, atol=NUMERICAL_TOLERANCE)
    
    def test_shannon_entropy_concentrated(
        self,
        phase3_validator: Phase3_ROBDDIsomorphismValidator,
        concentrated_distribution: np.ndarray,
    ) -> None:
        """Verifica entropía de Shannon para distribución concentrada."""
        entropy = phase3_validator._shannon_entropy_bits(concentrated_distribution)
        
        # H(X) = -0.9*log₂(0.9) - 0.1*log₂(0.1) ≈ 0.469 bits
        p = concentrated_distribution
        expected_entropy = float(-np.sum(p * np.log2(p)))
        
        assert_allclose(entropy, expected_entropy, atol=NUMERICAL_TOLERANCE)
    
    def test_min_entropy(
        self,
        phase3_validator: Phase3_ROBDDIsomorphismValidator,
        concentrated_distribution: np.ndarray,
    ) -> None:
        """Verifica min-entropía."""
        min_ent = phase3_validator._min_entropy(concentrated_distribution)
        
        # H_∞(X) = -log₂(max p) = -log₂(0.9) ≈ 0.152 bits
        expected = -math.log2(0.9)
        assert_allclose(min_ent, expected, atol=NUMERICAL_TOLERANCE)
    
    def test_collision_entropy(
        self,
        phase3_validator: Phase3_ROBDDIsomorphismValidator,
        uniform_distribution: np.ndarray,
    ) -> None:
        """Verifica entropía de colisión."""
        collision = phase3_validator._collision_entropy(uniform_distribution)
        
        # H₂(X) = -log₂(Σ p²) = -log₂(8 * (1/8)²) = log₂(8) = 3 bits
        expected = math.log2(8.0)
        assert_allclose(collision, expected, atol=NUMERICAL_TOLERANCE)
    
    def test_hartley_entropy(
        self,
        phase3_validator: Phase3_ROBDDIsomorphismValidator,
    ) -> None:
        """Verifica entropía de Hartley."""
        # Distribución con soporte de tamaño 4
        dist = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
        
        hartley = phase3_validator._hartley_entropy(dist)
        
        # H₀(X) = log₂(|support|) = log₂(4) = 2 bits
        expected = math.log2(4.0)
        assert_allclose(hartley, expected, atol=NUMERICAL_TOLERANCE)
    
    def test_entropy_analysis_complete(
        self,
        phase3_validator: Phase3_ROBDDIsomorphismValidator,
        uniform_distribution: np.ndarray,
    ) -> None:
        """Verifica análisis entrópico completo."""
        analysis = phase3_validator._compute_entropy_analysis(
            uniform_distribution,
            "test",
        )
        
        assert analysis.support_size == 8
        assert analysis.max_probability == 0.125
        assert_allclose(analysis.uniformity, 1.0, atol=NUMERICAL_TOLERANCE)
        
        # Todas las entropías deben ser 3 bits para distribución uniforme
        assert_allclose(analysis.shannon_entropy, 3.0, atol=NUMERICAL_TOLERANCE)
        assert_allclose(analysis.collision_entropy, 3.0, atol=NUMERICAL_TOLERANCE)
        assert_allclose(analysis.hartley_entropy, 3.0, atol=NUMERICAL_TOLERANCE)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   SUITE 10: PRUEBAS DE DIVERGENCIAS                                         ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class TestDivergenceComputations:
    """Pruebas de cómputo de divergencias."""
    
    def test_kl_divergence_identical_is_zero(
        self,
        phase3_validator: Phase3_ROBDDIsomorphismValidator,
        uniform_distribution: np.ndarray,
    ) -> None:
        """Verifica que D_KL(P‖P) = 0."""
        kl = phase3_validator._kl_divergence(
            uniform_distribution,
            uniform_distribution,
        )
        
        assert_allclose(kl, 0.0, atol=_KL_DIVERGENCE_TOLERANCE)
    
    def test_kl_divergence_non_negative(
        self,
        phase3_validator: Phase3_ROBDDIsomorphismValidator,
        uniform_distribution: np.ndarray,
        concentrated_distribution: np.ndarray,
    ) -> None:
        """Verifica que D_KL(P‖Q) ≥ 0."""
        # Padding para compatibilidad
        p = np.pad(uniform_distribution, (0, max(0, 8 - len(uniform_distribution))))
        q = np.pad(concentrated_distribution, (0, max(0, 8 - len(concentrated_distribution))))
        
        kl = phase3_validator._kl_divergence(p[:2], q[:2])
        
        assert kl >= -_KL_DIVERGENCE_TOLERANCE
    
    def test_jensen_shannon_symmetry(
        self,
        phase3_validator: Phase3_ROBDDIsomorphismValidator,
    ) -> None:
        """Verifica que JSD(P‖Q) = JSD(Q‖P)."""
        p = np.array([0.7, 0.3], dtype=np.float64)
        q = np.array([0.4, 0.6], dtype=np.float64)
        
        jsd_pq = phase3_validator._jensen_shannon_divergence(p, q)
        jsd_qp = phase3_validator._jensen_shannon_divergence(q, p)
        
        assert_allclose(jsd_pq, jsd_qp, atol=NUMERICAL_TOLERANCE)
    
    def test_jensen_shannon_bounds(
        self,
        phase3_validator: Phase3_ROBDDIsomorphismValidator,
        uniform_distribution: np.ndarray,
        concentrated_distribution: np.ndarray,
    ) -> None:
        """Verifica que 0 ≤ JSD(P‖Q) ≤ 1."""
        p = uniform_distribution[:2]
        q = concentrated_distribution[:2]
        
        jsd = phase3_validator._jensen_shannon_divergence(p, q)
        
        assert 0.0 <= jsd <= 1.0


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   SUITE 11: PRUEBAS DE FASE 3 (EQUIVALENCIA ROBDD)                          ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class TestPhase3_ROBDDValidation:
    """Pruebas de validación de equivalencia ROBDD."""
    
    def test_identical_distributions_accepted(
        self,
        phase3_validator: Phase3_ROBDDIsomorphismValidator,
        uniform_distribution: np.ndarray,
    ) -> None:
        """Verifica que distribuciones idénticas sean aceptadas."""
        audit = phase3_validator._validate_robdd_homotopy(
            uniform_distribution,
            uniform_distribution,
        )
        
        assert audit.is_homotopically_equivalent
        assert_allclose(audit.entropy_loss, 0.0, atol=_MIN_ENTROPY_TOLERANCE)
    
    def test_small_perturbation_accepted(
        self,
        phase3_validator: Phase3_ROBDDIsomorphismValidator,
        uniform_distribution: np.ndarray,
    ) -> None:
        """Verifica que pequeña perturbación sea aceptada."""
        # Perturbación muy pequeña
        perturbed = uniform_distribution.copy()
        perturbed[0] += 0.001
        perturbed[1] -= 0.001
        
        audit = phase3_validator._validate_robdd_homotopy(
            uniform_distribution,
            perturbed,
        )
        
        assert audit.is_homotopically_equivalent
    
    def test_large_divergence_rejected(
        self,
        phase3_validator: Phase3_ROBDDIsomorphismValidator,
        uniform_distribution: np.ndarray,
        concentrated_distribution: np.ndarray,
    ) -> None:
        """Verifica que gran divergencia sea rechazada."""
        # Padding para compatibilidad
        p_uniform = np.pad(uniform_distribution, (0, 0))
        p_concentrated = np.pad(concentrated_distribution, (0, 6))
        
        with pytest.raises(ROBDDHomotopyError, match="Ruptura homotópica"):
            phase3_validator._validate_robdd_homotopy(
                p_uniform,
                p_concentrated,
            )
    
    def test_entropy_analysis_stored(
        self,
        phase3_validator: Phase3_ROBDDIsomorphismValidator,
        uniform_distribution: np.ndarray,
    ) -> None:
        """Verifica que análisis entrópico se almacene."""
        audit = phase3_validator._validate_robdd_homotopy(
            uniform_distribution,
            uniform_distribution,
        )
        
        assert audit.original_entropy_analysis is not None
        assert audit.reduced_entropy_analysis is not None
        
        # Deben ser idénticos
        assert_allclose(
            audit.original_entropy_analysis.shannon_entropy,
            audit.reduced_entropy_analysis.shannon_entropy,
            atol=NUMERICAL_TOLERANCE,
        )
    
    def test_total_variation_distance_computed(
        self,
        phase3_validator: Phase3_ROBDDIsomorphismValidator,
    ) -> None:
        """Verifica cómputo de distancia de variación total."""
        p = np.array([0.5, 0.5], dtype=np.float64)
        q = np.array([0.3, 0.7], dtype=np.float64)
        
        audit = phase3_validator._validate_robdd_homotopy(p, q)
        
        # d_TV = 0.5 * Σ|p - q| = 0.5 * (0.2 + 0.2) = 0.2
        expected_tv = 0.2
        assert_allclose(
            audit.total_variation_distance,
            expected_tv,
            atol=NUMERICAL_TOLERANCE,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   SUITE 12: PRUEBAS DE COMPOSICIÓN FUNTORIAL COMPLETA                       ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class TestFunctorialComposition:
    """Pruebas de la composición funtorial completa Φ₃ ∘ Φ₂ ∘ Φ₁."""
    
    def test_full_pipeline_valid_minimization(
        self,
        mic_agent: MICMinimizerAgent,
        full_rank_matrix_gf2: np.ndarray,
        orthonormal_projection_2d: np.ndarray,
        uniform_distribution: np.ndarray,
    ) -> None:
        """Verifica el pipeline completo con minimización válida."""
        governance_state = mic_agent.execute_boolean_topology_governance(
            full_rank_matrix_gf2,
            orthonormal_projection_2d,
            uniform_distribution,
            uniform_distribution,
        )
        
        assert governance_state.is_topologically_valid
        assert governance_state.reduction_phase == BooleanReductionPhase.COMPLETE
        assert governance_state.grobner_audit.is_minimally_independent
        assert governance_state.unsat_core_audit.is_strictly_orthogonal
        assert governance_state.robdd_audit.is_homotopically_equivalent
    
    def test_full_pipeline_certificate_propagation(
        self,
        mic_agent: MICMinimizerAgent,
        identity_matrix_gf2: np.ndarray,
        orthonormal_projection_3d: np.ndarray,
        uniform_distribution: np.ndarray,
    ) -> None:
        """Verifica propagación de certificados entre fases."""
        governance_state = mic_agent.execute_boolean_topology_governance(
            identity_matrix_gf2,
            orthonormal_projection_3d,
            uniform_distribution,
            uniform_distribution,
        )
        
        # Verificar consistencia dimensional
        assert (
            governance_state.grobner_audit.cols
            == governance_state.unsat_core_audit.variable_dim
        )
    
    def test_full_pipeline_quality_score(
        self,
        mic_agent: MICMinimizerAgent,
        full_rank_matrix_gf2: np.ndarray,
        orthonormal_projection_2d: np.ndarray,
        uniform_distribution: np.ndarray,
    ) -> None:
        """Verifica cómputo de score de calidad global."""
        governance_state = mic_agent.execute_boolean_topology_governance(
            full_rank_matrix_gf2,
            orthonormal_projection_2d,
            uniform_distribution,
            uniform_distribution,
        )
        
        # Score debe estar en [0, 1]
        assert 0.0 <= governance_state.overall_quality_score <= 1.0
        
        # Para minimización perfecta, score debe ser alto
        assert governance_state.overall_quality_score >= 0.9
    
    def test_full_pipeline_risk_assessment(
        self,
        mic_agent: MICMinimizerAgent,
        full_rank_matrix_gf2: np.ndarray,
        orthonormal_projection_2d: np.ndarray,
        uniform_distribution: np.ndarray,
    ) -> None:
        """Verifica evaluación de riesgo."""
        governance_state = mic_agent.execute_boolean_topology_governance(
            full_rank_matrix_gf2,
            orthonormal_projection_2d,
            uniform_distribution,
            uniform_distribution,
        )
        
        assert governance_state.risk_assessment in ["NOMINAL", "WARNING", "CRITICAL"]
        assert governance_state.risk_assessment == "NOMINAL"
    
    def test_full_pipeline_grobner_violation_propagates(
        self,
        mic_agent: MICMinimizerAgent,
        rank_deficient_matrix_gf2: np.ndarray,
        orthonormal_projection_2d: np.ndarray,
        uniform_distribution: np.ndarray,
    ) -> None:
        """Verifica que violación de Gröbner aborte el pipeline."""
        with pytest.raises(GrobnerDegeneracyError):
            mic_agent.execute_boolean_topology_governance(
                rank_deficient_matrix_gf2,
                orthonormal_projection_2d,
                uniform_distribution,
                uniform_distribution,
            )
    
    def test_full_pipeline_orthogonality_violation_propagates(
        self,
        mic_agent: MICMinimizerAgent,
        full_rank_matrix_gf2: np.ndarray,
        non_orthogonal_projection: np.ndarray,
        uniform_distribution: np.ndarray,
    ) -> None:
        """Verifica que violación de ortogonalidad aborte el pipeline."""
        with pytest.raises(NonInterferenceViolationError):
            mic_agent.execute_boolean_topology_governance(
                full_rank_matrix_gf2,
                non_orthogonal_projection,
                uniform_distribution,
                uniform_distribution,
            )
    
    def test_full_pipeline_entropy_violation_propagates(
        self,
        mic_agent: MICMinimizerAgent,
        full_rank_matrix_gf2: np.ndarray,
        orthonormal_projection_2d: np.ndarray,
        uniform_distribution: np.ndarray,
        concentrated_distribution: np.ndarray,
    ) -> None:
        """Verifica que violación de entropía aborte el pipeline."""
        # Padding
        p_uniform = np.pad(uniform_distribution, (0, 0))
        p_concentrated = np.pad(concentrated_distribution, (0, 6))
        
        with pytest.raises(ROBDDHomotopyError):
            mic_agent.execute_boolean_topology_governance(
                full_rank_matrix_gf2,
                orthonormal_projection_2d,
                p_uniform,
                p_concentrated,
            )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   SUITE 13: PRUEBAS DE CASOS EXTREMOS                                       ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class TestEdgeCases:
    """Pruebas de casos extremos y límites."""
    
    def test_single_tool_single_variable(
        self,
        mic_agent: MICMinimizerAgent,
    ) -> None:
        """Verifica manejo de caso 1×1."""
        matrix_gf2 = np.array([[1]], dtype=np.uint8)
        projection = np.array([[1.0]], dtype=np.float64)
        dist = np.array([1.0], dtype=np.float64)
        
        governance_state = mic_agent.execute_boolean_topology_governance(
            matrix_gf2,
            projection,
            dist,
            dist,
        )
        
        assert governance_state.is_topologically_valid
    
    def test_large_hamming_code(
        self,
        mic_agent: MICMinimizerAgent,
        hamming_code_matrix: np.ndarray,
    ) -> None:
        """Verifica manejo de código de Hamming [7,4,3]."""
        # Proyección ortogonal correspondiente
        Q, _ = np.linalg.qr(np.random.randn(7, 4).T)
        projection = Q.T
        
        dist = np.ones(128, dtype=np.float64) / 128.0
        
        governance_state = mic_agent.execute_boolean_topology_governance(
            hamming_code_matrix,
            projection,
            dist,
            dist,
        )
        
        assert governance_state.is_topologically_valid
    
    def test_sparse_matrix(
        self,
        mic_agent: MICMinimizerAgent,
    ) -> None:
        """Verifica manejo de matriz dispersa."""
        # Matriz 5×10 muy dispersa
        sparse_matrix = np.zeros((5, 10), dtype=np.uint8)
        sparse_matrix[0, 0] = 1
        sparse_matrix[1, 3] = 1
        sparse_matrix[2, 5] = 1
        sparse_matrix[3, 7] = 1
        sparse_matrix[4, 9] = 1
        
        projection = np.eye(5, 10, dtype=np.float64)
        dist = np.ones(32, dtype=np.float64) / 32.0
        
        governance_state = mic_agent.execute_boolean_topology_governance(
            sparse_matrix,
            projection,
            dist,
            dist,
        )
        
        assert governance_state.is_topologically_valid
    
    def test_distribution_with_small_support(
        self,
        mic_agent: MICMinimizerAgent,
        identity_matrix_gf2: np.ndarray,
        orthonormal_projection_3d: np.ndarray,
    ) -> None:
        """Verifica manejo de distribución con soporte pequeño."""
        # Distribución con soporte de tamaño 2
        dist = np.array([0.5, 0.5], dtype=np.float64)
        
        governance_state = mic_agent.execute_boolean_topology_governance(
            identity_matrix_gf2,
            orthonormal_projection_3d,
            dist,
            dist,
        )
        
        assert governance_state.is_topologically_valid


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   SUITE 14: PRUEBAS BASADAS EN PROPIEDADES (HYPOTHESIS)                     ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class TestPropertyBased:
    """Pruebas basadas en propiedades usando Hypothesis."""
    
    @given(
        rows=st.integers(min_value=2, max_value=8),
        cols=st.integers(min_value=2, max_value=8),
    )
    @settings(max_examples=20, deadline=5000)
    def test_rank_nullity_theorem(
        self,
        phase1_auditor: Phase1_GrobnerBasisAuditor,
        rows: int,
        cols: int,
    ) -> None:
        """Propiedad: rank + nullity = cols (teorema del rango-nulidad)."""
        matrix = np.random.randint(0, 2, size=(rows, cols), dtype=np.uint8)
        
        props = phase1_auditor._compute_gf2_matrix_properties(matrix, "test")
        
        assert props.rank_gf2 + props.nullity_gf2 == cols
    
    @given(
        size=st.integers(min_value=2, max_value=16),
    )
    @settings(max_examples=15, deadline=5000)
    def test_shannon_entropy_bounds(
        self,
        phase3_validator: Phase3_ROBDDIsomorphismValidator,
        size: int,
    ) -> None:
        """Propiedad: 0 ≤ H(X) ≤ log₂(|X|)."""
        # Distribución aleatoria válida
        p = np.random.dirichlet(alpha=np.ones(size))
        
        entropy = phase3_validator._shannon_entropy_bits(p)
        
        assert 0.0 <= entropy <= math.log2(size) + NUMERICAL_TOLERANCE
    
    @given(
        size=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=15, deadline=5000)
    def test_kl_divergence_non_negative(
        self,
        phase3_validator: Phase3_ROBDDIsomorphismValidator,
        size: int,
    ) -> None:
        """Propiedad: D_KL(P‖Q) ≥ 0."""
        p = np.random.dirichlet(alpha=np.ones(size))
        q = np.random.dirichlet(alpha=np.ones(size))
        
        kl = phase3_validator._kl_divergence(p, q)
        
        assert kl >= -_KL_DIVERGENCE_TOLERANCE


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   SUITE 15: PRUEBAS DE ROBUSTEZ NUMÉRICA                                    ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class TestNumericalRobustness:
    """Pruebas de robustez numérica y estabilidad."""
    
    def test_orthogonality_perturbation_stability(
        self,
        phase2_certifier: Phase2_UnsatCoreCertifier,
        orthonormal_projection_3d: np.ndarray,
    ) -> None:
        """Verifica estabilidad ante pequeñas perturbaciones."""
        epsilon = 1e-11
        
        perturbation = np.random.randn(3, 3) * epsilon
        perturbation = (perturbation + perturbation.T) / 2.0  # Simétrica
        
        perturbed = orthonormal_projection_3d + perturbation
        
        # Debería seguir siendo aceptada
        audit = phase2_certifier._certify_non_interference_unsat(perturbed)
        
        assert audit.is_strictly_orthogonal
    
    def test_distribution_normalization_robustness(
        self,
        phase3_validator: Phase3_ROBDDIsomorphismValidator,
    ) -> None:
        """Verifica robustez de normalización de distribuciones."""
        # Distribución ligeramente no normalizada
        unnormalized = np.array([0.3, 0.3, 0.39], dtype=np.float64)
        
        normalized = phase3_validator._as_finite_probability_vector(
            "test",
            unnormalized,
            normalize=True,
        )
        
        assert_allclose(np.sum(normalized), 1.0, atol=_PROBABILITY_TOLERANCE)
    
    def test_gf2_arithmetic_stability(
        self,
        phase1_auditor: Phase1_GrobnerBasisAuditor,
    ) -> None:
        """Verifica estabilidad de aritmética GF(2)."""
        # Matriz con valores flotantes cercanos a enteros
        almost_binary = np.array([
            [1.0 + 1e-14, 0.0],
            [0.0, 1.0 - 1e-14],
        ], dtype=np.float64)
        
        matrix_gf2 = phase1_auditor._as_finite_gf2_matrix("test", almost_binary)
        
        expected = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        assert_array_equal(matrix_gf2, expected)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   SUITE 16: PRUEBAS DE BENCHMARKING Y RENDIMIENTO                           ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class TestPerformance:
    """Pruebas de rendimiento y benchmarking."""
    
    def test_benchmark_full_pipeline_small(
        self,
        benchmark,
        mic_agent: MICMinimizerAgent,
        identity_matrix_gf2: np.ndarray,
        orthonormal_projection_3d: np.ndarray,
        uniform_distribution: np.ndarray,
    ) -> None:
        """Benchmark del pipeline completo para dimensión pequeña."""
        result = benchmark(
            mic_agent.execute_boolean_topology_governance,
            identity_matrix_gf2,
            orthonormal_projection_3d,
            uniform_distribution[:8],
            uniform_distribution[:8],
        )
        
        assert result.is_topologically_valid
    
    def test_benchmark_gf2_rank(
        self,
        benchmark,
        phase1_auditor: Phase1_GrobnerBasisAuditor,
        random_gf2_matrix: Callable[[int, int], np.ndarray],
    ) -> None:
        """Benchmark de cómputo de rango GF(2)."""
        matrix = random_gf2_matrix(10, 10)
        
        rank, pivots = benchmark(
            phase1_auditor._gf2_rank,
            matrix,
        )
        
        assert 0 <= rank <= 10
    
    def test_benchmark_gram_matrix(
        self,
        benchmark,
        phase2_certifier: Phase2_UnsatCoreCertifier,
        random_projection_matrix: Callable[[int, int], np.ndarray],
    ) -> None:
        """Benchmark de cómputo de matriz de Gram."""
        projection = random_projection_matrix(5, 10, orthogonal=True)
        
        gram = benchmark(
            phase2_certifier._compute_gram_matrix,
            projection,
        )
        
        assert gram.shape == (5, 5)
    
    def test_benchmark_shannon_entropy(
        self,
        benchmark,
        phase3_validator: Phase3_ROBDDIsomorphismValidator,
        random_distribution: Callable[[int], np.ndarray],
    ) -> None:
        """Benchmark de cómputo de entropía de Shannon."""
        dist = random_distribution(256)
        
        entropy = benchmark(
            phase3_validator._shannon_entropy_bits,
            dist,
        )
        
        assert entropy >= 0.0


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   SUITE 17: PRUEBAS DE INTEGRACIÓN CON CASOS REALES                         ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class TestRealWorldScenarios:
    """Pruebas con escenarios realistas de uso."""
    
    def test_boolean_minimization_scenario(
        self,
        mic_agent: MICMinimizerAgent,
    ) -> None:
        """
        Simula minimización de funciones booleanas.
        
        Escenario:
            - 4 herramientas (funciones booleanas)
            - 8 variables booleanas
            - Minimización via ROBDD
        """
        # Matriz de funciones booleanas
        boolean_funcs = np.array([
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 1, 0, 0, 1, 1, 0],
            [1, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
        ], dtype=np.uint8)
        
        # Proyección ortogonal
        Q, _ = np.linalg.qr(np.random.randn(8, 4).T)
        projection = Q.T
        
        # Distribuciones
        dist_original = np.ones(256, dtype=np.float64) / 256.0
        dist_reduced = dist_original.copy()
        
        governance_state = mic_agent.execute_boolean_topology_governance(
            boolean_funcs,
            projection,
            dist_original,
            dist_reduced,
        )
        
        assert governance_state.is_topologically_valid
    
    def test_error_correction_code_scenario(
        self,
        mic_agent: MICMinimizerAgent,
        hamming_code_matrix: np.ndarray,
    ) -> None:
        """
        Simula verificación de código de corrección de errores.
        
        Escenario:
            - Código de Hamming [7,4,3]
            - Verificación de independencia de filas
            - Ortogonalidad de palabras código
        """
        # Proyección desde espacio de códigos
        Q, _ = np.linalg.qr(hamming_code_matrix.T)
        projection = Q.T
        
        # Distribución uniforme sobre palabras código
        dist = np.ones(16, dtype=np.float64) / 16.0
        
        governance_state = mic_agent.execute_boolean_topology_governance(
            hamming_code_matrix,
            projection,
            dist,
            dist,
        )
        
        assert governance_state.is_topologically_valid
        assert governance_state.grobner_audit.code_properties is not None
        assert governance_state.grobner_audit.code_properties.minimum_distance >= 3


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PYTEST
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--cov=app.agents.boole.tactics.mic_minimizer_agent",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--benchmark-only",
        "--benchmark-columns=min,max,mean,stddev",
    ])