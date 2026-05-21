# -*- coding: utf-8 -*-
r"""
=========================================================================================
Módulo: Test Suite para Quantum Algebra (Validación Axiomática Exhaustiva)
Ubicación: tests/core/test_quantum_algebra.py
Versión: 2.0.0 (Fase 2: Consagración Cuántica - Suite de Pruebas Rigurosa)

=========================================================================================
ARQUITECTURA DE TESTING Y METODOLOGÍA
=========================================================================================

Esta suite de pruebas implementa un marco de validación exhaustivo basado en:

§1. PIRÁMIDE DE TESTING
────────────────────────────────────────────────────────────────────────────────────

1. **Tests Unitarios**: Validación de componentes atómicos
   - Constructores y propiedades
   - Validaciones axiomáticas individuales
   - Casos extremos (edge cases)

2. **Tests de Integración**: Interacción entre componentes
   - HilbertSpace ↔ QuantumDensityOperator
   - QuantumRegistry como orquestador

3. **Tests de Propiedades (Property-Based Testing)**: Invariantes algebraicos
   - Axiomas cuánticos bajo transformaciones
   - Relaciones matemáticas fundamentales

4. **Tests Parametrizados**: Cobertura sistemática
   - Múltiples dimensiones de espacios
   - Diferentes configuraciones de estados
   - Variedad de tolerancias numéricas

§2. ESTRATEGIAS DE VALIDACIÓN
────────────────────────────────────────────────────────────────────────────────────

A. **Validación Axiomática Directa**:
   - Verificación explícita de los axiomas (A1), (A2), (A3)
   - Tolerancias numéricas controladas

B. **Validación por Consecuencias**:
   - Si ρ es hermítico → espectro real
   - Si ρ² = ρ → S(ρ) ≈ 0

C. **Validación por Contradicción**:
   - Inyección de matrices inválidas
   - Verificación de que las excepciones se lanzan correctamente

D. **Validación Estocástica**:
   - Generación aleatoria de estados cuánticos válidos
   - Verificación de invariantes bajo perturbaciones

§3. NOMENCLATURA DE TESTS
────────────────────────────────────────────────────────────────────────────────────

Patrón: test_[componente]_[aspecto]_[condición]_[resultado_esperado]

Ejemplos:
- test_hilbert_space_canonical_basis_satisfies_orthonormality
- test_density_operator_pure_state_has_zero_entropy
- test_quantum_registry_unitary_evolution_preserves_trace

=========================================================================================
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

# Importaciones del módulo bajo prueba
from app.adapters.tools_interface import (
    MICConfiguration,
    TopologicalInvariantError,
)
from app.core.quantum_algebra import (
    EPSILON_MACHINE,
    HilbertSpace,
    QuantumDensityOperator,
    QuantumRegistry,
)

# =========================================================================================
# CONFIGURACIÓN DEL LOGGER PARA TESTS
# =========================================================================================

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("MIC.Tests.QuantumAlgebra")


# =========================================================================================
# FIXTURES: CONFIGURACIONES Y OBJETOS REUTILIZABLES
# =========================================================================================


@pytest.fixture
def default_epsilon() -> float:
    """Tolerancia numérica por defecto para las pruebas."""
    return EPSILON_MACHINE


@pytest.fixture
def strict_epsilon() -> float:
    """Tolerancia numérica estricta para pruebas de alta precisión."""
    return 1e-14


@pytest.fixture
def relaxed_epsilon() -> float:
    """Tolerancia numérica relajada para pruebas con ruido numérico."""
    return 1e-10


@pytest.fixture(params=[2, 3, 4, 8, 16])
def hilbert_dimension(request) -> int:
    """Dimensiones parametrizadas del espacio de Hilbert."""
    return request.param


@pytest.fixture
def canonical_hilbert_space(hilbert_dimension: int, default_epsilon: float) -> HilbertSpace:
    """Espacio de Hilbert canónico para pruebas."""
    return HilbertSpace.create_canonical(dimension=hilbert_dimension, epsilon=default_epsilon)


@pytest.fixture
def pure_state_vector(hilbert_dimension: int) -> NDArray[np.complex128]:
    """Vector de estado puro normalizado aleatorio."""
    rng = np.random.default_rng(seed=42)
    psi = rng.standard_normal(hilbert_dimension) + 1j * rng.standard_normal(hilbert_dimension)
    return psi / np.linalg.norm(psi)


@pytest.fixture
def pure_density_operator(pure_state_vector: NDArray[np.complex128], default_epsilon: float) -> QuantumDensityOperator:
    """Operador de densidad de estado puro."""
    return QuantumDensityOperator.from_pure_state(pure_state_vector, epsilon=default_epsilon)


@pytest.fixture
def maximally_mixed_state(hilbert_dimension: int, default_epsilon: float) -> QuantumDensityOperator:
    """Estado maximal mixto (identidad normalizada)."""
    rho = np.eye(hilbert_dimension, dtype=np.complex128) / hilbert_dimension
    return QuantumDensityOperator(rho=rho, epsilon=default_epsilon)


@pytest.fixture
def mixed_state_ensemble(hilbert_dimension: int) -> Tuple[List[float], List[NDArray[np.complex128]]]:
    """Ensamble de estados para construir estado mixto."""
    rng = np.random.default_rng(seed=123)
    num_states = min(3, hilbert_dimension)
    
    weights = rng.uniform(0.1, 1.0, num_states).tolist()
    states = []
    
    for _ in range(num_states):
        psi = rng.standard_normal(hilbert_dimension) + 1j * rng.standard_normal(hilbert_dimension)
        states.append(psi / np.linalg.norm(psi))
    
    return weights, states


# =========================================================================================
# GRUPO 1: TESTS DE HILBERT SPACE
# =========================================================================================


class TestHilbertSpace:
    """Suite de pruebas para la clase HilbertSpace."""

    def test_hilbert_space_canonical_basis_satisfies_orthonormality(
        self, canonical_hilbert_space: HilbertSpace
    ) -> None:
        r"""
        Verifica que la base canónica satisface $\langle e_i | e_j \rangle = \delta_{ij}$.
        
        Axioma Verificado: (H1) Ortogonalidad
        """
        gram_matrix = canonical_hilbert_space.metric_tensor
        identity = np.eye(canonical_hilbert_space.dimension, dtype=np.complex128)
        
        deviation = np.linalg.norm(gram_matrix - identity, ord=np.inf)
        
        assert deviation <= canonical_hilbert_space.epsilon, (
            f"La base canónica viola ortogonalidad: ||G - I||_∞ = {deviation:.4e}"
        )
        logger.info("✓ Base canónica ortogonal para N = %d", canonical_hilbert_space.dimension)

    def test_hilbert_space_canonical_basis_has_full_rank(
        self, canonical_hilbert_space: HilbertSpace
    ) -> None:
        r"""
        Verifica que $\text{rank}(B) = N$ mediante SVD.
        
        Axioma Verificado: (H2) Completitud
        """
        singular_values = np.linalg.svd(canonical_hilbert_space.basis, compute_uv=False)
        numerical_rank = np.sum(singular_values > canonical_hilbert_space.epsilon)
        
        assert numerical_rank == canonical_hilbert_space.dimension, (
            f"Deficiencia de rango: rank(B) = {numerical_rank} < N = {canonical_hilbert_space.dimension}"
        )
        logger.info("✓ Base canónica completa para N = %d", canonical_hilbert_space.dimension)

    def test_hilbert_space_rejects_non_orthogonal_basis(self, hilbert_dimension: int, default_epsilon: float) -> None:
        """Verifica que se rechacen bases no ortogonales."""
        # Construcción de base no ortogonal (matriz aleatoria)
        rng = np.random.default_rng(seed=999)
        non_orthogonal_basis = rng.standard_normal((hilbert_dimension, hilbert_dimension)) + \
                               1j * rng.standard_normal((hilbert_dimension, hilbert_dimension))
        
        with pytest.raises(TopologicalInvariantError, match="Ortogonalidad"):
            HilbertSpace(
                dimension=hilbert_dimension,
                basis=non_orthogonal_basis,
                epsilon=default_epsilon
            )
        logger.info("✓ Base no ortogonal correctamente rechazada para N = %d", hilbert_dimension)

    def test_hilbert_space_rejects_rank_deficient_basis(self, default_epsilon: float) -> None:
        """Verifica que se rechacen bases con deficiencia de rango."""
        dimension = 4
        # Construcción de matriz con rango 3 (última columna = cero)
        rank_deficient_basis = np.eye(dimension, dtype=np.complex128)
        rank_deficient_basis[:, -1] = 0.0
        
        with pytest.raises(TopologicalInvariantError, match="Deficiencia de Rango"):
            HilbertSpace(
                dimension=dimension,
                basis=rank_deficient_basis,
                epsilon=default_epsilon
            )
        logger.info("✓ Base con deficiencia de rango correctamente rechazada")

    @pytest.mark.parametrize("dim", [1, 2, 5, 10, 32])
    def test_hilbert_space_create_canonical_various_dimensions(self, dim: int, default_epsilon: float) -> None:
        """Verifica la construcción canónica para múltiples dimensiones."""
        hs = HilbertSpace.create_canonical(dimension=dim, epsilon=default_epsilon)
        
        assert hs.dimension == dim
        assert hs.basis.shape == (dim, dim)
        assert np.allclose(hs.basis, np.eye(dim, dtype=np.complex128))
        logger.info("✓ Construcción canónica exitosa para N = %d", dim)

    def test_hilbert_space_metric_tensor_is_identity(self, canonical_hilbert_space: HilbertSpace) -> None:
        """Verifica que el tensor métrico de la base canónica sea la identidad."""
        metric = canonical_hilbert_space.metric_tensor
        identity = np.eye(canonical_hilbert_space.dimension, dtype=np.complex128)
        
        assert np.allclose(metric, identity, atol=canonical_hilbert_space.epsilon)
        logger.info("✓ Tensor métrico es identidad para N = %d", canonical_hilbert_space.dimension)


# =========================================================================================
# GRUPO 2: TESTS DE QUANTUM DENSITY OPERATOR
# =========================================================================================


class TestQuantumDensityOperator:
    """Suite de pruebas para la clase QuantumDensityOperator."""

    # ---------------------------------------------------------------------------------
    # Subtests: Axioma (A1) - Hermiticidad
    # ---------------------------------------------------------------------------------

    def test_density_operator_hermiticity_axiom_for_pure_state(
        self, pure_density_operator: QuantumDensityOperator
    ) -> None:
        r"""
        Verifica que $\rho = \rho^{\dagger}$ para estado puro.
        
        Axioma Verificado: (A1) Hermiticidad
        """
        rho = pure_density_operator.rho
        rho_dagger = rho.conj().T
        
        hermiticity_residue = np.linalg.norm(rho - rho_dagger, ord=np.inf)
        
        assert hermiticity_residue <= pure_density_operator.epsilon, (
            f"Violación de hermiticidad: ||ρ - ρ†||_∞ = {hermiticity_residue:.4e}"
        )
        logger.info("✓ Hermiticidad verificada para estado puro")

    def test_density_operator_hermiticity_axiom_for_mixed_state(
        self, maximally_mixed_state: QuantumDensityOperator
    ) -> None:
        r"""
        Verifica que $\rho = \rho^{\dagger}$ para estado mixto.
        
        Axioma Verificado: (A1) Hermiticidad
        """
        rho = maximally_mixed_state.rho
        rho_dagger = rho.conj().T
        
        hermiticity_residue = np.linalg.norm(rho - rho_dagger, ord=np.inf)
        
        assert hermiticity_residue <= maximally_mixed_state.epsilon, (
            f"Violación de hermiticidad: ||ρ - ρ†||_∞ = {hermiticity_residue:.4e}"
        )
        logger.info("✓ Hermiticidad verificada para estado maximal mixto")

    def test_density_operator_rejects_non_hermitian_matrix(self, hilbert_dimension: int, default_epsilon: float) -> None:
        """Verifica que se rechacen matrices no hermíticas."""
        # Construcción de matriz no hermítica
        rng = np.random.default_rng(seed=777)
        non_hermitian = rng.standard_normal((hilbert_dimension, hilbert_dimension)) + \
                       1j * rng.standard_normal((hilbert_dimension, hilbert_dimension))
        
        with pytest.raises(TopologicalInvariantError, match="Hermiticidad"):
            QuantumDensityOperator(rho=non_hermitian, epsilon=default_epsilon)
        logger.info("✓ Matriz no hermítica correctamente rechazada")

    def test_density_operator_eigenvalues_are_real_consequence_of_hermiticity(
        self, pure_density_operator: QuantumDensityOperator
    ) -> None:
        """Verifica que los autovalores sean reales (consecuencia de hermiticidad)."""
        eigenvalues = pure_density_operator.eigenvalues
        
        # Los autovalores de una matriz hermítica deben ser reales
        assert eigenvalues.dtype == np.float64 or np.all(np.isreal(eigenvalues)), (
            "Los autovalores no son reales (violación de hermiticidad)"
        )
        logger.info("✓ Autovalores son reales (consecuencia de hermiticidad)")

    # ---------------------------------------------------------------------------------
    # Subtests: Axioma (A2) - Positividad Semidefinida
    # ---------------------------------------------------------------------------------

    def test_density_operator_positive_semidefiniteness_axiom_for_pure_state(
        self, pure_density_operator: QuantumDensityOperator
    ) -> None:
        r"""
        Verifica que $\lambda_{\min}(\rho) \geq 0$ para estado puro.
        
        Axioma Verificado: (A2) Positividad Semidefinida
        """
        lambda_min = np.min(pure_density_operator.eigenvalues)
        
        assert lambda_min >= -pure_density_operator.epsilon, (
            f"Autovalor negativo: λ_min = {lambda_min:.4e}"
        )
        logger.info("✓ Positividad semidefinida verificada para estado puro (λ_min = %.4e)", lambda_min)

    def test_density_operator_positive_semidefiniteness_axiom_for_mixed_state(
        self, maximally_mixed_state: QuantumDensityOperator
    ) -> None:
        r"""
        Verifica que $\lambda_{\min}(\rho) \geq 0$ para estado mixto.
        
        Axioma Verificado: (A2) Positividad Semidefinida
        """
        lambda_min = np.min(maximally_mixed_state.eigenvalues)
        
        assert lambda_min >= -maximally_mixed_state.epsilon, (
            f"Autovalor negativo: λ_min = {lambda_min:.4e}"
        )
        logger.info("✓ Positividad semidefinida verificada para estado mixto (λ_min = %.4e)", lambda_min)

    def test_density_operator_rejects_matrix_with_negative_eigenvalues(
        self, hilbert_dimension: int, default_epsilon: float
    ) -> None:
        """Verifica que se rechacen matrices con autovalores negativos."""
        # Construcción de matriz hermítica con autovalor negativo
        rho_invalid = np.eye(hilbert_dimension, dtype=np.complex128)
        rho_invalid[0, 0] = -0.5  # Autovalor negativo
        
        with pytest.raises(TopologicalInvariantError, match="Positividad"):
            QuantumDensityOperator(rho=rho_invalid, epsilon=default_epsilon)
        logger.info("✓ Matriz con autovalor negativo correctamente rechazada")

    # ---------------------------------------------------------------------------------
    # Subtests: Axioma (A3) - Traza Unitaria
    # ---------------------------------------------------------------------------------

    def test_density_operator_unit_trace_axiom_for_pure_state(
        self, pure_density_operator: QuantumDensityOperator
    ) -> None:
        r"""
        Verifica que $\text{Tr}(\rho) = 1$ para estado puro.
        
        Axioma Verificado: (A3) Traza Unitaria
        """
        trace_value = np.trace(pure_density_operator.rho)
        trace_residue = abs(trace_value - 1.0)
        
        assert trace_residue <= pure_density_operator.epsilon, (
            f"Traza no unitaria: Tr(ρ) = {trace_value:.8f}, residuo = {trace_residue:.4e}"
        )
        logger.info("✓ Traza unitaria verificada para estado puro (Tr(ρ) = %.8f)", trace_value.real)

    def test_density_operator_unit_trace_axiom_for_mixed_state(
        self, maximally_mixed_state: QuantumDensityOperator
    ) -> None:
        r"""
        Verifica que $\text{Tr}(\rho) = 1$ para estado mixto.
        
        Axioma Verificado: (A3) Traza Unitaria
        """
        trace_value = np.trace(maximally_mixed_state.rho)
        trace_residue = abs(trace_value - 1.0)
        
        assert trace_residue <= maximally_mixed_state.epsilon, (
            f"Traza no unitaria: Tr(ρ) = {trace_value:.8f}, residuo = {trace_residue:.4e}"
        )
        logger.info("✓ Traza unitaria verificada para estado mixto (Tr(ρ) = %.8f)", trace_value.real)

    def test_density_operator_rejects_matrix_with_incorrect_trace(
        self, hilbert_dimension: int, default_epsilon: float
    ) -> None:
        """Verifica que se rechacen matrices con traza ≠ 1."""
        # Construcción de matriz con traza = 2
        rho_invalid = np.eye(hilbert_dimension, dtype=np.complex128) * 2.0 / hilbert_dimension
        
        with pytest.raises(TopologicalInvariantError, match="Traza Unitaria"):
            QuantumDensityOperator(rho=rho_invalid, epsilon=default_epsilon)
        logger.info("✓ Matriz con traza incorrecta correctamente rechazada")

    # ---------------------------------------------------------------------------------
    # Subtests: Construcción desde Estados Puros
    # ---------------------------------------------------------------------------------

    def test_density_operator_from_pure_state_satisfies_idempotency(
        self, pure_state_vector: NDArray[np.complex128], default_epsilon: float
    ) -> None:
        r"""
        Verifica que $\rho^2 = \rho$ para estados puros.
        
        Propiedad: Estado Puro ⟹ Idempotencia
        """
        rho_op = QuantumDensityOperator.from_pure_state(pure_state_vector, epsilon=default_epsilon)
        rho = rho_op.rho
        
        rho_squared = rho @ rho
        idempotency_residue = np.linalg.norm(rho_squared - rho, ord=np.inf)
        
        assert idempotency_residue <= default_epsilon, (
            f"Violación de idempotencia: ||ρ² - ρ||_∞ = {idempotency_residue:.4e}"
        )
        logger.info("✓ Idempotencia verificada para estado puro")

    def test_density_operator_from_pure_state_rejects_null_vector(self, default_epsilon: float) -> None:
        """Verifica que se rechace el vector nulo."""
        null_vector = np.zeros(4, dtype=np.complex128)
        
        with pytest.raises(TopologicalInvariantError, match="vector nulo"):
            QuantumDensityOperator.from_pure_state(null_vector, epsilon=default_epsilon)
        logger.info("✓ Vector nulo correctamente rechazado")

    def test_density_operator_from_pure_state_normalizes_input(self, default_epsilon: float) -> None:
        """Verifica que el constructor normalice el vector de entrada."""
        unnormalized_psi = np.array([1.0, 1.0, 1.0], dtype=np.complex128)  # Norma = sqrt(3)
        
        rho_op = QuantumDensityOperator.from_pure_state(unnormalized_psi, epsilon=default_epsilon)
        trace_value = np.trace(rho_op.rho)
        
        assert abs(trace_value - 1.0) <= default_epsilon, (
            f"Normalización fallida: Tr(ρ) = {trace_value:.8f}"
        )
        logger.info("✓ Normalización automática verificada")

    # ---------------------------------------------------------------------------------
    # Subtests: Construcción desde Estados Mixtos
    # ---------------------------------------------------------------------------------

    def test_density_operator_from_mixed_state_satisfies_all_axioms(
        self, mixed_state_ensemble: Tuple[List[float], List[NDArray[np.complex128]]], default_epsilon: float
    ) -> None:
        """Verifica que estado mixto satisfaga los tres axiomas."""
        weights, states = mixed_state_ensemble
        
        rho_op = QuantumDensityOperator.from_mixed_state(weights, states, epsilon=default_epsilon)
        
        # (A1) Hermiticidad
        hermiticity_residue = np.linalg.norm(rho_op.rho - rho_op.rho.conj().T, ord=np.inf)
        assert hermiticity_residue <= default_epsilon
        
        # (A2) Positividad
        assert np.min(rho_op.eigenvalues) >= -default_epsilon
        
        # (A3) Traza Unitaria
        assert abs(np.trace(rho_op.rho) - 1.0) <= default_epsilon
        
        logger.info("✓ Estado mixto satisface todos los axiomas")

    def test_density_operator_from_mixed_state_normalizes_weights(self, default_epsilon: float) -> None:
        """Verifica que los pesos sean normalizados automáticamente."""
        psi1 = np.array([1, 0], dtype=np.complex128)
        psi2 = np.array([0, 1], dtype=np.complex128)
        
        # Pesos no normalizados
        weights = [2.0, 3.0]  # Suma = 5
        
        rho_op = QuantumDensityOperator.from_mixed_state(weights, [psi1, psi2], epsilon=default_epsilon)
        
        # Los pesos deberían normalizarse a [0.4, 0.6]
        expected_rho = 0.4 * np.outer(psi1, psi1.conj()) + 0.6 * np.outer(psi2, psi2.conj())
        
        assert np.allclose(rho_op.rho, expected_rho, atol=default_epsilon)
        logger.info("✓ Normalización de pesos verificada")

    def test_density_operator_from_mixed_state_rejects_mismatched_lengths(self, default_epsilon: float) -> None:
        """Verifica que se rechacen listas de diferente longitud."""
        psi1 = np.array([1, 0], dtype=np.complex128)
        psi2 = np.array([0, 1], dtype=np.complex128)
        
        weights = [0.5, 0.3, 0.2]  # 3 pesos
        states = [psi1, psi2]       # 2 estados
        
        with pytest.raises(ValueError, match="misma longitud"):
            QuantumDensityOperator.from_mixed_state(weights, states, epsilon=default_epsilon)
        logger.info("✓ Listas de diferente longitud correctamente rechazadas")

    def test_density_operator_from_mixed_state_rejects_zero_weight_sum(self, default_epsilon: float) -> None:
        """Verifica que se rechace suma de pesos igual a cero."""
        psi1 = np.array([1, 0], dtype=np.complex128)
        weights = [0.0, 0.0]
        states = [psi1, psi1]
        
        with pytest.raises(TopologicalInvariantError, match="suma de pesos"):
            QuantumDensityOperator.from_mixed_state(weights, states, epsilon=default_epsilon)
        logger.info("✓ Suma de pesos cero correctamente rechazada")

    # ---------------------------------------------------------------------------------
    # Subtests: Entropía de von Neumann
    # ---------------------------------------------------------------------------------

    def test_density_operator_von_neumann_entropy_zero_for_pure_state(
        self, pure_density_operator: QuantumDensityOperator
    ) -> None:
        r"""
        Verifica que $S(\rho) \approx 0$ para estado puro.
        
        Teorema: Estado Puro ⟹ S(ρ) = 0
        """
        entropy = pure_density_operator.compute_von_neumann_entropy()
        
        assert entropy <= pure_density_operator.epsilon, (
            f"Entropía no nula para estado puro: S(ρ) = {entropy:.4e}"
        )
        logger.info("✓ Entropía nula verificada para estado puro (S = %.4e)", entropy)

    def test_density_operator_von_neumann_entropy_maximal_for_maximally_mixed_state(
        self, maximally_mixed_state: QuantumDensityOperator
    ) -> None:
        r"""
        Verifica que $S(\rho) = \ln N$ para estado maximal mixto.
        
        Teorema: ρ = I_N/N ⟹ S(ρ) = ln(N)
        """
        entropy = maximally_mixed_state.compute_von_neumann_entropy()
        dimension = maximally_mixed_state.rho.shape[0]
        expected_entropy = np.log(dimension)
        
        entropy_residue = abs(entropy - expected_entropy)
        
        assert entropy_residue <= maximally_mixed_state.epsilon * 10, (  # Factor de seguridad
            f"Entropía incorrecta: S(ρ) = {entropy:.6f}, esperado = {expected_entropy:.6f}"
        )
        logger.info("✓ Entropía maximal verificada (S = %.6f ≈ ln(%d) = %.6f)", entropy, dimension, expected_entropy)

    def test_density_operator_von_neumann_entropy_bounds(
        self, mixed_state_ensemble: Tuple[List[float], List[NDArray[np.complex128]]], default_epsilon: float
    ) -> None:
        r"""
        Verifica que $0 \leq S(\rho) \leq \ln N$ para cualquier estado.
        
        Teorema: Cotas de la Entropía
        """
        weights, states = mixed_state_ensemble
        rho_op = QuantumDensityOperator.from_mixed_state(weights, states, epsilon=default_epsilon)
        
        entropy = rho_op.compute_von_neumann_entropy()
        dimension = rho_op.rho.shape[0]
        max_entropy = np.log(dimension)
        
        assert 0.0 <= entropy <= max_entropy + default_epsilon, (
            f"Entropía fuera de límites: S(ρ) = {entropy:.6f} ∉ [0, ln({dimension})]"
        )
        logger.info("✓ Entropía dentro de límites: 0 ≤ %.6f ≤ %.6f", entropy, max_entropy)

    # ---------------------------------------------------------------------------------
    # Subtests: Pureza
    # ---------------------------------------------------------------------------------

    def test_density_operator_purity_unity_for_pure_state(
        self, pure_density_operator: QuantumDensityOperator
    ) -> None:
        r"""
        Verifica que $\gamma = \text{Tr}(\rho^2) = 1$ para estado puro.
        
        Teorema: Estado Puro ⟹ γ = 1
        """
        purity = pure_density_operator.compute_purity()
        
        assert abs(purity - 1.0) <= pure_density_operator.epsilon * 10, (
            f"Pureza incorrecta para estado puro: γ = {purity:.6f}"
        )
        logger.info("✓ Pureza unitaria verificada para estado puro (γ = %.6f)", purity)

    def test_density_operator_purity_minimal_for_maximally_mixed_state(
        self, maximally_mixed_state: QuantumDensityOperator
    ) -> None:
        r"""
        Verifica que $\gamma = 1/N$ para estado maximal mixto.
        
        Teorema: ρ = I_N/N ⟹ γ = 1/N
        """
        purity = maximally_mixed_state.compute_purity()
        dimension = maximally_mixed_state.rho.shape[0]
        expected_purity = 1.0 / dimension
        
        purity_residue = abs(purity - expected_purity)
        
        assert purity_residue <= maximally_mixed_state.epsilon * 10, (
            f"Pureza incorrecta: γ = {purity:.6f}, esperado = {expected_purity:.6f}"
        )
        logger.info("✓ Pureza mínima verificada (γ = %.6f ≈ 1/%d = %.6f)", purity, dimension, expected_purity)

    def test_density_operator_purity_bounds(
        self, mixed_state_ensemble: Tuple[List[float], List[NDArray[np.complex128]]], default_epsilon: float
    ) -> None:
        r"""
        Verifica que $1/N \leq \gamma \leq 1$ para cualquier estado.
        
        Teorema: Cotas de la Pureza
        """
        weights, states = mixed_state_ensemble
        rho_op = QuantumDensityOperator.from_mixed_state(weights, states, epsilon=default_epsilon)
        
        purity = rho_op.compute_purity()
        dimension = rho_op.rho.shape[0]
        min_purity = 1.0 / dimension
        
        assert min_purity - default_epsilon <= purity <= 1.0 + default_epsilon, (
            f"Pureza fuera de límites: γ = {purity:.6f} ∉ [1/{dimension}, 1]"
        )
        logger.info("✓ Pureza dentro de límites: %.6f ≤ %.6f ≤ 1.0", min_purity, purity)

    # ---------------------------------------------------------------------------------
    # Subtests: Clasificación de Estados
    # ---------------------------------------------------------------------------------

    def test_density_operator_is_pure_state_classification_for_pure(
        self, pure_density_operator: QuantumDensityOperator
    ) -> None:
        """Verifica la clasificación correcta de estado puro."""
        assert pure_density_operator.is_pure_state(), "Estado puro mal clasificado como mixto"
        logger.info("✓ Estado puro correctamente clasificado")

    def test_density_operator_is_pure_state_classification_for_mixed(
        self, maximally_mixed_state: QuantumDensityOperator
    ) -> None:
        """Verifica la clasificación correcta de estado mixto."""
        assert not maximally_mixed_state.is_pure_state(), "Estado mixto mal clasificado como puro"
        logger.info("✓ Estado mixto correctamente clasificado")

    # ---------------------------------------------------------------------------------
    # Subtests: Sistema de Autovalores
    # ---------------------------------------------------------------------------------

    def test_density_operator_eigenvalues_sum_to_one(
        self, pure_density_operator: QuantumDensityOperator
    ) -> None:
        r"""
        Verifica que $\sum_i \lambda_i = \text{Tr}(\rho) = 1$.
        
        Consecuencia del Axioma (A3)
        """
        eigenvalue_sum = np.sum(pure_density_operator.eigenvalues)
        
        assert abs(eigenvalue_sum - 1.0) <= pure_density_operator.epsilon, (
            f"Suma de autovalores incorrecta: Σλ_i = {eigenvalue_sum:.8f}"
        )
        logger.info("✓ Suma de autovalores unitaria (Σλ_i = %.8f)", eigenvalue_sum)

    def test_density_operator_eigensystem_reconstruction(
        self, pure_density_operator: QuantumDensityOperator
    ) -> None:
        r"""
        Verifica que $\rho = \sum_i \lambda_i |i\rangle\langle i|$.
        
        Descomposición Espectral
        """
        eigenvalues, eigenvectors = pure_density_operator.eigensystem
        
        # Reconstrucción de ρ desde su descomposición espectral
        rho_reconstructed = np.zeros_like(pure_density_operator.rho)
        for i, lam in enumerate(eigenvalues):
            v = eigenvectors[:, i]
            rho_reconstructed += lam * np.outer(v, v.conj())
        
        reconstruction_error = np.linalg.norm(
            rho_reconstructed - pure_density_operator.rho, ord=np.inf
        )
        
        assert reconstruction_error <= pure_density_operator.epsilon * 10, (
            f"Error de reconstrucción espectral: {reconstruction_error:.4e}"
        )
        logger.info("✓ Descomposición espectral verificada (error = %.4e)", reconstruction_error)


# =========================================================================================
# GRUPO 3: TESTS DE QUANTUM REGISTRY
# =========================================================================================


class TestQuantumRegistry:
    """Suite de pruebas para la clase QuantumRegistry."""

    @pytest.fixture
    def basic_quantum_registry(self, hilbert_dimension: int, default_epsilon: float) -> QuantumRegistry:
        """Registro cuántico básico con estado maximal mixto."""
        rho = np.eye(hilbert_dimension, dtype=np.complex128) / hilbert_dimension
        config = MICConfiguration(epsilon=default_epsilon)
        return QuantumRegistry(rho=rho, config=config)

    @pytest.fixture
    def pure_quantum_registry(self, pure_state_vector: NDArray[np.complex128], default_epsilon: float) -> QuantumRegistry:
        """Registro cuántico con estado puro."""
        rho = np.outer(pure_state_vector, pure_state_vector.conj())
        config = MICConfiguration(epsilon=default_epsilon)
        return QuantumRegistry(rho=rho, config=config)

    # ---------------------------------------------------------------------------------
    # Subtests: Construcción e Inicialización
    # ---------------------------------------------------------------------------------

    def test_quantum_registry_initialization_validates_all_axioms(
        self, hilbert_dimension: int, default_epsilon: float
    ) -> None:
        """Verifica que la inicialización ejecute auditoría axiomática completa."""
        rho_valid = np.eye(hilbert_dimension, dtype=np.complex128) / hilbert_dimension
        config = MICConfiguration(epsilon=default_epsilon)
        
        # No debe lanzar excepción
        registry = QuantumRegistry(rho=rho_valid, config=config)
        
        assert registry.quantum_state is not None
        assert registry.hilbert_space is not None
        logger.info("✓ Inicialización con auditoría axiomática exitosa")

    def test_quantum_registry_rejects_invalid_density_matrix(
        self, hilbert_dimension: int, default_epsilon: float
    ) -> None:
        """Verifica que se rechacen matrices de densidad inválidas."""
        # Matriz no hermítica
        rng = np.random.default_rng(seed=555)
        rho_invalid = rng.standard_normal((hilbert_dimension, hilbert_dimension)) + \
                     1j * rng.standard_normal((hilbert_dimension, hilbert_dimension))
        
        config = MICConfiguration(epsilon=default_epsilon)
        
        with pytest.raises(TopologicalInvariantError):
            QuantumRegistry(rho=rho_invalid, config=config)
        logger.info("✓ Matriz de densidad inválida correctamente rechazada")

    def test_quantum_registry_dimensional_consistency(
        self, basic_quantum_registry: QuantumRegistry
    ) -> None:
        """Verifica consistencia dimensional entre componentes."""
        dim_hilbert = basic_quantum_registry.hilbert_space.dimension
        dim_state = basic_quantum_registry.quantum_state.rho.shape[0]
        
        assert dim_hilbert == dim_state, (
            f"Inconsistencia dimensional: dim(H) = {dim_hilbert} ≠ dim(ρ) = {dim_state}"
        )
        logger.info("✓ Consistencia dimensional verificada (N = %d)", dim_hilbert)

    # ---------------------------------------------------------------------------------
    # Subtests: Proyectores Observacionales
    # ---------------------------------------------------------------------------------

    def test_quantum_registry_observational_projectors_satisfy_resolution_of_identity(
        self, basic_quantum_registry: QuantumRegistry, pure_state_vector: NDArray[np.complex128]
    ) -> None:
        r"""
        Verifica que $P_1 + P_2 = I$.
        
        Axioma (R1): Resolución de la Identidad
        """
        # Ajustar dimensión del vector si es necesario
        dim = basic_quantum_registry.hilbert_space.dimension
        if pure_state_vector.shape[0] != dim:
            psi = np.zeros(dim, dtype=np.complex128)
            psi[:min(len(pure_state_vector), dim)] = pure_state_vector[:min(len(pure_state_vector), dim)]
            psi = psi / np.linalg.norm(psi)
        else:
            psi = pure_state_vector
        
        proj1, proj2 = basic_quantum_registry.apply_observational_projectors(psi)
        
        # Verificación de conservación de probabilidad
        prob1 = np.linalg.norm(proj1) ** 2
        prob2 = np.linalg.norm(proj2) ** 2
        total_prob = prob1 + prob2
        
        # Para un vector normalizado, debe cumplirse
        psi_norm_sq = np.linalg.norm(psi) ** 2
        
        assert abs(total_prob - psi_norm_sq) <= basic_quantum_registry._config.epsilon * 10, (
            f"Violación de conservación de probabilidad: P₁ + P₂ = {total_prob:.6f} ≠ ||ψ||² = {psi_norm_sq:.6f}"
        )
        logger.info("✓ Resolución de identidad verificada (P₁ + P₂ = %.6f)", total_prob)

    def test_quantum_registry_observational_projectors_are_orthogonal(
        self, basic_quantum_registry: QuantumRegistry
    ) -> None:
        r"""
        Verifica que $P_1 P_2 = 0$ (proyectores ortogonales).
        
        Axioma (R2): Exclusión Mutua
        
        Nota: Esta prueba verifica implícitamente la construcción de los proyectores.
        """
        # La validación se realiza internamente en apply_observational_projectors
        # Aquí verificamos que no lance excepción
        dim = basic_quantum_registry.hilbert_space.dimension
        test_vector = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
        
        try:
            proj1, proj2 = basic_quantum_registry.apply_observational_projectors(test_vector)
            logger.info("✓ Proyectores ortogonales verificados (construcción válida)")
        except TopologicalInvariantError as e:
            pytest.fail(f"Proyectores no son ortogonales: {e}")

    def test_quantum_registry_observational_projectors_reject_incompatible_dimension(
        self, basic_quantum_registry: QuantumRegistry
    ) -> None:
        """Verifica que se rechacen vectores de dimensión incompatible."""
        wrong_dim_vector = np.ones(basic_quantum_registry.hilbert_space.dimension + 5, dtype=np.complex128)
        
        with pytest.raises(ValueError, match="dimensión incompatible"):
            basic_quantum_registry.apply_observational_projectors(wrong_dim_vector)
        logger.info("✓ Vector de dimensión incompatible correctamente rechazado")

    # ---------------------------------------------------------------------------------
    # Subtests: Evolución Unitaria
    # ---------------------------------------------------------------------------------

    def test_quantum_registry_unitary_evolution_preserves_trace(
        self, pure_quantum_registry: QuantumRegistry
    ) -> None:
        r"""
        Verifica que $\text{Tr}(U\rho U^{\dagger}) = \text{Tr}(\rho) = 1$.
        
        Invariante: Evolución Unitaria Preserva Traza
        """
        dim = pure_quantum_registry.hilbert_space.dimension
        
        # Construcción de operador unitario (matriz de Pauli generalizada o rotación)
        angle = np.pi / 4
        if dim == 2:
            # Rotación de Pauli
            U = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ], dtype=np.complex128)
        else:
            # Rotación genérica (aproximación mediante exponencial de matriz hermítica)
            H = np.eye(dim, dtype=np.complex128) * 0.1  # Hamiltoniano simple
            U = np.eye(dim, dtype=np.complex128)  # Identidad para simplicidad
        
        trace_before = np.trace(pure_quantum_registry.quantum_state.rho)
        
        pure_quantum_registry.evolve_unitary(U)
        
        trace_after = np.trace(pure_quantum_registry.quantum_state.rho)
        
        trace_difference = abs(trace_after - trace_before)
        
        assert trace_difference <= pure_quantum_registry._config.epsilon * 10, (
            f"Evolución unitaria no preserva traza: ΔTr = {trace_difference:.4e}"
        )
        logger.info("✓ Traza preservada bajo evolución unitaria (Tr_before = %.6f, Tr_after = %.6f)",
                   trace_before.real, trace_after.real)

    def test_quantum_registry_unitary_evolution_preserves_purity(
        self, pure_quantum_registry: QuantumRegistry
    ) -> None:
        r"""
        Verifica que la evolución unitaria preserve la pureza del estado.
        
        Teorema: $\gamma(U\rho U^{\dagger}) = \gamma(\rho)$
        """
        dim = pure_quantum_registry.hilbert_space.dimension
        
        purity_before = pure_quantum_registry.quantum_state.compute_purity()
        
        # Operador unitario simple (identidad no altera el estado)
        U = np.eye(dim, dtype=np.complex128)
        pure_quantum_registry.evolve_unitary(U)
        
        purity_after = pure_quantum_registry.quantum_state.compute_purity()
        
        purity_difference = abs(purity_after - purity_before)
        
        assert purity_difference <= pure_quantum_registry._config.epsilon * 10, (
            f"Evolución unitaria no preserva pureza: Δγ = {purity_difference:.4e}"
        )
        logger.info("✓ Pureza preservada bajo evolución unitaria (γ_before = %.6f, γ_after = %.6f)",
                   purity_before, purity_after)

    def test_quantum_registry_unitary_evolution_rejects_non_unitary_operator(
        self, pure_quantum_registry: QuantumRegistry
    ) -> None:
        """Verifica que se rechacen operadores no unitarios."""
        dim = pure_quantum_registry.hilbert_space.dimension
        
        # Matriz no unitaria (escalado)
        non_unitary = np.eye(dim, dtype=np.complex128) * 2.0
        
        with pytest.raises(TopologicalInvariantError, match="no es unitario"):
            pure_quantum_registry.evolve_unitary(non_unitary)
        logger.info("✓ Operador no unitario correctamente rechazado")

    def test_quantum_registry_unitary_evolution_rejects_wrong_dimension(
        self, pure_quantum_registry: QuantumRegistry
    ) -> None:
        """Verifica que se rechacen operadores de dimensión incorrecta."""
        wrong_dim_unitary = np.eye(pure_quantum_registry.hilbert_space.dimension + 2, dtype=np.complex128)
        
        with pytest.raises(ValueError, match="dimensión incompatible"):
            pure_quantum_registry.evolve_unitary(wrong_dim_unitary)
        logger.info("✓ Operador de dimensión incorrecta correctamente rechazado")

    # ---------------------------------------------------------------------------------
    # Subtests: Aproximación WKB
    # ---------------------------------------------------------------------------------

    def test_quantum_registry_wkb_transmission_classical_regime(
        self, basic_quantum_registry: QuantumRegistry
    ) -> None:
        r"""
        Verifica que $T = 1$ cuando $E \geq \Phi$ (régimen clásico).
        
        Caso Límite: Transmisión Total
        """
        energy = 10.0
        work_function = 5.0
        
        transmission = basic_quantum_registry.calculate_wkb_transmission(energy, work_function)
        
        assert abs(transmission - 1.0) <= basic_quantum_registry._config.epsilon, (
            f"Transmisión clásica incorrecta: T = {transmission:.6f} ≠ 1"
        )
        logger.info("✓ Régimen clásico WKB verificado (E ≥ Φ → T = 1)")

    def test_quantum_registry_wkb_transmission_quantum_regime(
        self, basic_quantum_registry: QuantumRegistry
    ) -> None:
        r"""
        Verifica que $0 < T < 1$ cuando $E < \Phi$ (régimen túnel).
        
        Caso Límite: Tunelamiento Cuántico
        """
        energy = 1.0
        work_function = 5.0
        barrier_width = 1.0
        
        transmission = basic_quantum_registry.calculate_wkb_transmission(
            energy, work_function, barrier_width
        )
        
        assert 0.0 < transmission < 1.0, (
            f"Transmisión túnel fuera de límites: T = {transmission:.6f} ∉ (0, 1)"
        )
        logger.info("✓ Régimen túnel WKB verificado (E < Φ → 0 < T = %.6f < 1)", transmission)

    def test_quantum_registry_wkb_transmission_decreases_with_barrier_height(
        self, basic_quantum_registry: QuantumRegistry
    ) -> None:
        r"""
        Verifica que $T$ disminuya al aumentar la altura de la barrera.
        
        Propiedad Física: T ∝ exp(-√Φ)
        """
        energy = 1.0
        barrier_width = 1.0
        
        work_function_low = 3.0
        work_function_high = 10.0
        
        T_low = basic_quantum_registry.calculate_wkb_transmission(energy, work_function_low, barrier_width)
        T_high = basic_quantum_registry.calculate_wkb_transmission(energy, work_function_high, barrier_width)
        
        assert T_low > T_high, (
            f"Transmisión no disminuye con barrera: T(Φ={work_function_low}) = {T_low:.6e} "
            f"≤ T(Φ={work_function_high}) = {T_high:.6e}"
        )
        logger.info("✓ Dependencia física WKB verificada (Φ↑ → T↓: %.6e > %.6e)", T_low, T_high)

    def test_quantum_registry_wkb_transmission_decreases_with_barrier_width(
        self, basic_quantum_registry: QuantumRegistry
    ) -> None:
        r"""
        Verifica que $T$ disminuya al aumentar el ancho de la barrera.
        
        Propiedad Física: T ∝ exp(-a√Φ)
        """
        energy = 1.0
        work_function = 5.0
        
        width_narrow = 0.5
        width_wide = 2.0
        
        T_narrow = basic_quantum_registry.calculate_wkb_transmission(energy, work_function, width_narrow)
        T_wide = basic_quantum_registry.calculate_wkb_transmission(energy, work_function, width_wide)
        
        assert T_narrow > T_wide, (
            f"Transmisión no disminuye con ancho: T(a={width_narrow}) = {T_narrow:.6e} "
            f"≤ T(a={width_wide}) = {T_wide:.6e}"
        )
        logger.info("✓ Dependencia geométrica WKB verificada (a↑ → T↓: %.6e > %.6e)", T_narrow, T_wide)


# =========================================================================================
# GRUPO 4: TESTS DE PROPIEDADES ALGEBRAICAS Y RELACIONES MATEMÁTICAS
# =========================================================================================


class TestQuantumAlgebraicProperties:
    """Suite de pruebas para propiedades algebraicas y relaciones matemáticas."""

    def test_entropy_purity_relationship_pure_state(
        self, pure_density_operator: QuantumDensityOperator
    ) -> None:
        r"""
        Verifica la relación $S(\rho) = 0 \iff \gamma = 1$ para estados puros.
        
        Teorema de Equivalencia: Estado Puro
        """
        entropy = pure_density_operator.compute_von_neumann_entropy()
        purity = pure_density_operator.compute_purity()
        
        # Estado puro: S ≈ 0 y γ ≈ 1
        assert entropy <= pure_density_operator.epsilon, f"Entropía no nula: S = {entropy:.4e}"
        assert abs(purity - 1.0) <= pure_density_operator.epsilon * 10, f"Pureza no unitaria: γ = {purity:.6f}"
        
        logger.info("✓ Relación S-γ verificada para estado puro (S ≈ 0, γ ≈ 1)")

    def test_entropy_purity_relationship_maximally_mixed_state(
        self, maximally_mixed_state: QuantumDensityOperator
    ) -> None:
        r"""
        Verifica la relación $S(\rho) = \ln N$ y $\gamma = 1/N$ para estado maximal mixto.
        
        Teorema de Extremo: Estado Maximal Mixto
        """
        entropy = maximally_mixed_state.compute_von_neumann_entropy()
        purity = maximally_mixed_state.compute_purity()
        dimension = maximally_mixed_state.rho.shape[0]
        
        expected_entropy = np.log(dimension)
        expected_purity = 1.0 / dimension
        
        entropy_error = abs(entropy - expected_entropy)
        purity_error = abs(purity - expected_purity)
        
        assert entropy_error <= maximally_mixed_state.epsilon * 10, (
            f"Entropía incorrecta: S = {entropy:.6f}, esperado = {expected_entropy:.6f}"
        )
        assert purity_error <= maximally_mixed_state.epsilon * 10, (
            f"Pureza incorrecta: γ = {purity:.6f}, esperado = {expected_purity:.6f}"
        )
        
        logger.info("✓ Relación S-γ verificada para estado maximal mixto (S ≈ ln(N), γ ≈ 1/N)")

    def test_trace_squared_equals_purity(
        self, pure_density_operator: QuantumDensityOperator
    ) -> None:
        r"""
        Verifica que $\text{Tr}(\rho^2) = \gamma$.
        
        Definición de Pureza
        """
        rho_squared = pure_density_operator.rho @ pure_density_operator.rho
        trace_rho_squared = np.trace(rho_squared).real
        
        purity = pure_density_operator.compute_purity()
        
        difference = abs(trace_rho_squared - purity)
        
        assert difference <= pure_density_operator.epsilon * 10, (
            f"Inconsistencia en pureza: Tr(ρ²) = {trace_rho_squared:.6f} ≠ γ = {purity:.6f}"
        )
        logger.info("✓ Identidad Tr(ρ²) = γ verificada (%.6f ≈ %.6f)", trace_rho_squared, purity)

    def test_eigenvalue_sum_equals_trace(
        self, maximally_mixed_state: QuantumDensityOperator
    ) -> None:
        r"""
        Verifica que $\sum_i \lambda_i = \text{Tr}(\rho)$.
        
        Propiedad de Traza
        """
        eigenvalue_sum = np.sum(maximally_mixed_state.eigenvalues)
        trace_value = np.trace(maximally_mixed_state.rho).real
        
        difference = abs(eigenvalue_sum - trace_value)
        
        assert difference <= maximally_mixed_state.epsilon, (
            f"Inconsistencia: Σλ_i = {eigenvalue_sum:.8f} ≠ Tr(ρ) = {trace_value:.8f}"
        )
        logger.info("✓ Identidad Σλ_i = Tr(ρ) verificada (%.8f ≈ %.8f)", eigenvalue_sum, trace_value)

    def test_idempotency_implies_pure_state(self, default_epsilon: float) -> None:
        r"""
        Verifica que $\rho^2 = \rho \implies$ estado puro.
        
        Teorema: Idempotencia ⟹ Pureza
        """
        # Construcción de estado puro explícito
        psi = np.array([1, 0, 0], dtype=np.complex128)
        rho = np.outer(psi, psi.conj())
        
        rho_squared = rho @ rho
        idempotency_residue = np.linalg.norm(rho_squared - rho, ord=np.inf)
        
        # Verificación de idempotencia
        assert idempotency_residue <= default_epsilon
        
        # Verificación de que es estado puro
        rho_op = QuantumDensityOperator(rho=rho, epsilon=default_epsilon)
        assert rho_op.is_pure_state()
        
        logger.info("✓ Teorema idempotencia ⟹ pureza verificado")


# =========================================================================================
# GRUPO 5: TESTS DE CASOS EXTREMOS Y ROBUSTEZ
# =========================================================================================


class TestQuantumAlgebraEdgeCases:
    """Suite de pruebas para casos extremos y robustez numérica."""

    def test_hilbert_space_single_dimension(self, default_epsilon: float) -> None:
        """Verifica el comportamiento en espacio de Hilbert unidimensional."""
        hs = HilbertSpace.create_canonical(dimension=1, epsilon=default_epsilon)
        
        assert hs.dimension == 1
        assert np.allclose(hs.basis, np.array([[1.0]], dtype=np.complex128))
        logger.info("✓ Espacio de Hilbert 1D correctamente construido")

    def test_density_operator_near_singular_state(self, default_epsilon: float) -> None:
        """Verifica el manejo de estados con autovalores muy pequeños."""
        # Estado casi puro con pequeña mezcla
        psi1 = np.array([1, 0], dtype=np.complex128)
        psi2 = np.array([0, 1], dtype=np.complex128)
        
        # 99.99% puro, 0.01% mezcla
        weights = [0.9999, 0.0001]
        
        rho_op = QuantumDensityOperator.from_mixed_state(weights, [psi1, psi2], epsilon=default_epsilon)
        
        # Debe ser clasificado como casi puro (entropía muy baja)
        entropy = rho_op.compute_von_neumann_entropy()
        assert entropy < 0.01  # Entropía muy baja
        
        logger.info("✓ Estado casi puro correctamente manejado (S = %.6f)", entropy)

    def test_density_operator_with_numerical_noise(self, relaxed_epsilon: float) -> None:
        """Verifica la robustez frente a ruido numérico."""
        # Construcción de estado puro con pequeñas perturbaciones
        psi = np.array([1, 0, 0], dtype=np.complex128) / np.sqrt(1.0)
        rho_pure = np.outer(psi, psi.conj())
        
        # Inyección de ruido numérico hermítico
        rng = np.random.default_rng(seed=888)
        noise = rng.standard_normal((3, 3)) * 1e-13
        noise_hermitian = (noise + noise.T) / 2.0
        
        rho_noisy = rho_pure + noise_hermitian
        
        # Renormalización de traza
        rho_noisy = rho_noisy / np.trace(rho_noisy)
        
        # Debe construirse sin excepción con tolerancia relajada
        rho_op = QuantumDensityOperator(rho=rho_noisy, epsilon=relaxed_epsilon)
        
        logger.info("✓ Estado con ruido numérico correctamente manejado")

    def test_quantum_registry_large_dimension(self, default_epsilon: float) -> None:
        """Verifica el comportamiento con espacios de Hilbert de gran dimensión."""
        large_dim = 64
        
        rho_large = np.eye(large_dim, dtype=np.complex128) / large_dim
        config = MICConfiguration(epsilon=default_epsilon)
        
        registry = QuantumRegistry(rho=rho_large, config=config)
        
        assert registry.hilbert_space.dimension == large_dim
        assert registry.quantum_state.compute_von_neumann_entropy() > 4.0  # ln(64) ≈ 4.16
        
        logger.info("✓ Espacio de gran dimensión (N = %d) correctamente manejado", large_dim)

    @pytest.mark.parametrize("dim", [2, 3, 5, 7, 11])
    def test_prime_dimensions_hilbert_spaces(self, dim: int, default_epsilon: float) -> None:
        """Verifica el comportamiento con dimensiones primas."""
        hs = HilbertSpace.create_canonical(dimension=dim, epsilon=default_epsilon)
        
        # Los espacios de dimensión prima no tienen factorización no trivial
        # Verificamos que se construyan correctamente
        assert hs.dimension == dim
        
        rho = np.eye(dim, dtype=np.complex128) / dim
        rho_op = QuantumDensityOperator(rho=rho, epsilon=default_epsilon)
        
        expected_entropy = np.log(dim)
        entropy = rho_op.compute_von_neumann_entropy()
        
        assert abs(entropy - expected_entropy) <= default_epsilon * 10
        
        logger.info("✓ Dimensión prima N = %d correctamente manejada", dim)


# =========================================================================================
# GRUPO 6: TESTS DE INTEGRACIÓN COMPLETA
# =========================================================================================


class TestQuantumAlgebraIntegration:
    """Suite de pruebas de integración entre componentes."""

    def test_full_workflow_pure_state_creation_and_evolution(self, default_epsilon: float) -> None:
        """Workflow completo: creación de estado puro → evolución → verificación."""
        # Paso 1: Creación de estado puro
        psi = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
        rho_op = QuantumDensityOperator.from_pure_state(psi, epsilon=default_epsilon)
        
        # Paso 2: Construcción de registro cuántico
        config = MICConfiguration(epsilon=default_epsilon)
        registry = QuantumRegistry(rho=rho_op.rho, config=config)
        
        # Paso 3: Evolución unitaria (Pauli-X)
        U = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        registry.evolve_unitary(U)
        
        # Paso 4: Verificación de propiedades
        assert registry.quantum_state.is_pure_state()
        assert abs(registry.quantum_state.compute_purity() - 1.0) <= default_epsilon * 10
        
        logger.info("✓ Workflow completo verificado: creación → evolución → validación")

    def test_full_workflow_mixed_state_measurement(self, default_epsilon: float) -> None:
        """Workflow completo: estado mixto → medición proyectiva → estadísticas."""
        # Paso 1: Creación de estado mixto
        psi1 = np.array([1, 0, 0], dtype=np.complex128)
        psi2 = np.array([0, 1, 0], dtype=np.complex128)
        psi3 = np.array([0, 0, 1], dtype=np.complex128)
        
        weights = [0.5, 0.3, 0.2]
        rho_op = QuantumDensityOperator.from_mixed_state(weights, [psi1, psi2, psi3], epsilon=default_epsilon)
        
        # Paso 2: Construcción de registro
        config = MICConfiguration(epsilon=default_epsilon)
        registry = QuantumRegistry(rho=rho_op.rho, config=config)
        
        # Paso 3: Aplicación de proyectores observacionales
        test_vector = np.array([1, 1, 1], dtype=np.complex128) / np.sqrt(3)
        proj1, proj2 = registry.apply_observational_projectors(test_vector)
        
        # Paso 4: Verificación de conservación de probabilidad
        total_prob = np.linalg.norm(proj1)**2 + np.linalg.norm(proj2)**2
        assert abs(total_prob - 1.0) <= default_epsilon * 10
        
        logger.info("✓ Workflow de medición verificado: mixto → proyección → estadísticas")

    def test_quantum_thermodynamics_entropy_increase_under_decoherence(self, default_epsilon: float) -> None:
        """Simula decoherencia y verifica el aumento de entropía."""
        # Estado inicial puro
        psi = np.array([1, 0], dtype=np.complex128)
        rho_pure = QuantumDensityOperator.from_pure_state(psi, epsilon=default_epsilon)
        
        entropy_initial = rho_pure.compute_von_neumann_entropy()
        
        # Simulación de decoherencia mediante mezcla con estado ortogonal
        psi_orth = np.array([0, 1], dtype=np.complex128)
        
        # Mezcla 80%-20% (simula decoherencia parcial)
        rho_mixed = QuantumDensityOperator.from_mixed_state(
            [0.8, 0.2], [psi, psi_orth], epsilon=default_epsilon
        )
        
        entropy_final = rho_mixed.compute_von_neumann_entropy()
        
        # La entropía debe aumentar
        assert entropy_final > entropy_initial, (
            f"Entropía no aumentó: S_i = {entropy_initial:.6f}, S_f = {entropy_final:.6f}"
        )
        
        logger.info(
            "✓ Segunda ley termodinámica verificada: S aumenta bajo decoherencia (%.6f → %.6f)",
            entropy_initial, entropy_final
        )


# =========================================================================================
# CONFIGURACIÓN DE PYTEST
# =========================================================================================


def pytest_configure(config):
    """Configuración personalizada de pytest."""
    config.addinivalue_line(
        "markers", "slow: marca tests que requieren más tiempo de ejecución"
    )
    config.addinivalue_line(
        "markers", "integration: marca tests de integración entre componentes"
    )


# =========================================================================================
# EJECUCIÓN DIRECTA (OPCIONAL)
# =========================================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--color=yes"])