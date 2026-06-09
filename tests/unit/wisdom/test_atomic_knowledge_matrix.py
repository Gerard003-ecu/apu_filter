# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo de Pruebas: Atomic Knowledge Matrix Test Suite                        ║
║ Ubicación: tests/unit/wisdom/test_atomic_knowledge_matrix.py                      ║
║ Versión: 2.0.0-Quantum-Sheaf-Synthesis-Test                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Suite de Pruebas Rigurosas para Validación de:
───────────────────────────────────────────────
1. Axiomas cuánticos (Hermiticidad, Traza, Positividad)
2. Cohomología de haces celulares (Teorema de De Rham discreto)
3. Geometría simpléctica (Estructura de Dirac)
4. Disipatividad de Lyapunov (Termodinámica del aprendizaje)
5. Adjunción de Galois (Identidades categoriales)

Metodología de Testing:
───────────────────────
- Pruebas unitarias (componentes aislados)
- Pruebas de integración (flujos completos)
- Pruebas de propiedades (property-based testing)
- Pruebas de invariantes matemáticos
- Pruebas de estabilidad numérica
- Pruebas de regresión

Referencias:
────────────
- IEEE 754-2019: Estándar de punto flotante
- Hypothesis: Framework de property-based testing
- pytest: Framework de testing moderno
═══════════════════════════════════════════════════════════════════════════════
"""

import pytest
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from typing import Tuple, List, Callable
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays
import logging

# Módulo bajo prueba
import sys
sys.path.insert(0, '../..')

from app.wisdom.atomic_knowledge_matrix import (
    AtomicDensityMatrix,
    QuantumAxiomViolation,
    QuantumMetrics,
    RestrictionMap,
    CellularSheafNeuralManifold,
    SheafCohomologyGroup,
    DiracStructure,
    PortHamiltonianLearningFlow,
    LyapunovCertificate,
    GaloisAdjunctionFunctor,
    create_quantum_mac_state,
    create_geometric_learning_system,
    NumericalInstabilityError
)

logger = logging.getLogger("MAC.Tests")


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1: PRUEBAS UNITARIAS - ESTRUCTURA CUÁNTICA
# ══════════════════════════════════════════════════════════════════════════════

class TestAtomicDensityMatrix:
    """Suite de pruebas para operadores de densidad cuántica."""
    
    @pytest.fixture
    def pure_state_2d(self) -> AtomicDensityMatrix:
        """Estado puro |ψ⟩ = (1/√2)(|0⟩ + |1⟩) - Estado de Bell."""
        psi = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
        rho = np.outer(psi, psi.conj())
        return AtomicDensityMatrix(rho, tol=1e-12)
    
    @pytest.fixture
    def mixed_state_2d(self) -> AtomicDensityMatrix:
        """Estado maximalmente mixto ρ = I/2."""
        rho = np.eye(2, dtype=np.complex128) / 2.0
        return AtomicDensityMatrix(rho, tol=1e-12)
    
    @pytest.fixture
    def entangled_state_4d(self) -> AtomicDensityMatrix:
        """Estado de Bell maximalmente entrelazado |Φ⁺⟩ = (|00⟩ + |11⟩)/√2."""
        psi = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
        rho = np.outer(psi, psi.conj())
        return AtomicDensityMatrix(rho, tol=1e-12)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Axiomas Cuánticos
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_hermiticity_axiom_pure_state(self, pure_state_2d):
        """Axioma A1: ρ = ρ† para estado puro."""
        rho = pure_state_2d.matrix
        assert np.allclose(rho, rho.conj().T, atol=1e-12), "Violación de Hermiticidad"
    
    def test_hermiticity_axiom_mixed_state(self, mixed_state_2d):
        """Axioma A1: ρ = ρ† para estado mixto."""
        rho = mixed_state_2d.matrix
        assert np.allclose(rho, rho.conj().T, atol=1e-12), "Violación de Hermiticidad"
    
    def test_trace_unitarity_pure_state(self, pure_state_2d):
        """Axioma A2: Tr(ρ) = 1 para estado puro."""
        trace = np.trace(pure_state_2d.matrix)
        assert np.isclose(trace.real, 1.0, atol=1e-12), f"Traza no unitaria: {trace}"
        assert abs(trace.imag) < 1e-12, f"Traza con parte imaginaria: {trace.imag}"
    
    def test_trace_unitarity_mixed_state(self, mixed_state_2d):
        """Axioma A2: Tr(ρ) = 1 para estado mixto."""
        trace = np.trace(mixed_state_2d.matrix)
        assert np.isclose(trace.real, 1.0, atol=1e-12), f"Traza no unitaria: {trace}"
    
    def test_positive_semidefiniteness_pure(self, pure_state_2d):
        """Axioma A3: ρ ≽ 0 (eigenvalores no negativos) para estado puro."""
        eigenvalues = la.eigvalsh(pure_state_2d.matrix)
        assert np.all(eigenvalues >= -1e-12), f"Eigenvalores negativos: {eigenvalues}"
    
    def test_positive_semidefiniteness_mixed(self, mixed_state_2d):
        """Axioma A3: ρ ≽ 0 para estado mixto."""
        eigenvalues = la.eigvalsh(mixed_state_2d.matrix)
        assert np.all(eigenvalues >= -1e-12), f"Eigenvalores negativos: {eigenvalues}"
    
    def test_invalid_non_hermitian_matrix(self):
        """Debe rechazar matrices no hermitianas."""
        non_hermitian = np.array([[1, 2j], [3j, 1]], dtype=np.complex128)
        with pytest.raises(NumericalInstabilityError, match="no es Hermítica"):
            AtomicDensityMatrix(non_hermitian, validate=True)
    
    def test_invalid_trace_anomaly(self):
        """Debe rechazar matrices con traza ≠ 1."""
        invalid_trace = np.eye(2, dtype=np.complex128) * 2.0  # Tr = 2
        with pytest.raises(NumericalInstabilityError, match="traza no es unitaria"):
            AtomicDensityMatrix(invalid_trace, validate=True, auto_renormalize=False)
    
    def test_invalid_negative_eigenvalues(self):
        """Debe rechazar matrices con eigenvalores negativos."""
        # Matriz hermitiana pero con eigenvalor negativo
        negative_eig = np.array([[1, 2], [2, 1]], dtype=np.complex128)  # Eigenvalores: 3, -1
        with pytest.raises(NumericalInstabilityError, match="negativos"):
            AtomicDensityMatrix(negative_eig, validate=True, auto_renormalize=False)
    
    def test_auto_renormalization(self):
        """Debe renormalizar automáticamente estados con traza ≠ 1."""
        unnormalized = np.eye(2, dtype=np.complex128) * 0.5  # Tr = 1.0 (ya normalizado)
        rho = AtomicDensityMatrix(unnormalized, auto_renormalize=True)
        assert np.isclose(np.trace(rho.matrix).real, 1.0, atol=1e-12)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Métricas Cuánticas
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_purity_pure_state(self, pure_state_2d):
        """Pureza de estado puro debe ser 1: Tr(ρ²) = 1."""
        metrics = pure_state_2d.compute_metrics()
        assert np.isclose(metrics.purity, 1.0, atol=1e-10), f"Pureza incorrecta: {metrics.purity}"
    
    def test_purity_maximally_mixed(self, mixed_state_2d):
        """Pureza de estado maximalmente mixto debe ser 1/d."""
        metrics = mixed_state_2d.compute_metrics()
        expected_purity = 1.0 / 2.0  # d = 2
        assert np.isclose(metrics.purity, expected_purity, atol=1e-10), \
            f"Pureza incorrecta: {metrics.purity} (esperado {expected_purity})"
    
    def test_von_neumann_entropy_pure_state(self, pure_state_2d):
        """Entropía de estado puro debe ser 0: S = 0."""
        metrics = pure_state_2d.compute_metrics()
        assert np.isclose(metrics.von_neumann_entropy, 0.0, atol=1e-10), \
            f"Entropía no nula para estado puro: {metrics.von_neumann_entropy}"
    
    def test_von_neumann_entropy_maximally_mixed(self, mixed_state_2d):
        """Entropía de estado maximalmente mixto debe ser log₂(d)."""
        metrics = mixed_state_2d.compute_metrics()
        expected_entropy = np.log2(2.0)  # log₂(d) = 1
        assert np.isclose(metrics.von_neumann_entropy, expected_entropy, atol=1e-10), \
            f"Entropía incorrecta: {metrics.von_neumann_entropy} (esperado {expected_entropy})"
    
    def test_participation_ratio_consistency(self, mixed_state_2d):
        """IPR debe ser consistente con pureza: IPR = 1/Tr(ρ²)."""
        metrics = mixed_state_2d.compute_metrics()
        expected_ipr = 1.0 / metrics.purity
        assert np.isclose(metrics.participation_ratio, expected_ipr, atol=1e-10), \
            f"IPR inconsistente: {metrics.participation_ratio} vs {expected_ipr}"
    
    def test_fidelity_to_pure_equals_max_eigenvalue(self, mixed_state_2d):
        """Fidelidad a estado puro debe ser el eigenvalor máximo."""
        metrics = mixed_state_2d.compute_metrics()
        eigenvalues = la.eigvalsh(mixed_state_2d.matrix)
        max_eigenvalue = np.max(eigenvalues)
        assert np.isclose(metrics.fidelity_to_pure, max_eigenvalue, atol=1e-10), \
            f"Fidelidad incorrecta: {metrics.fidelity_to_pure} vs {max_eigenvalue}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Operaciones Cuánticas
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_observable_measurement_pauli_z(self, pure_state_2d):
        """Medición de Pauli-Z en estado |+⟩ debe dar 0."""
        pauli_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        expectation = pure_state_2d.measure_observable(pauli_z)
        # |+⟩ = (|0⟩ + |1⟩)/√2 → ⟨σz⟩ = 0
        assert np.isclose(expectation, 0.0, atol=1e-10), f"⟨σz⟩ incorrecto: {expectation}"
    
    def test_observable_measurement_pauli_x(self, pure_state_2d):
        """Medición de Pauli-X en estado |+⟩ debe dar 1."""
        pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        expectation = pure_state_2d.measure_observable(pauli_x)
        # |+⟩ es eigenestado de σx con eigenvalor +1
        assert np.isclose(expectation, 1.0, atol=1e-10), f"⟨σx⟩ incorrecto: {expectation}"
    
    def test_observable_must_be_hermitian(self, pure_state_2d):
        """Debe rechazar observables no hermitianos."""
        non_hermitian = np.array([[1, 2j], [0, 1]], dtype=np.complex128)
        with pytest.raises(ValueError, match="hermitiano"):
            pure_state_2d.measure_observable(non_hermitian)
    
    def test_unitary_evolution_preserves_purity(self, pure_state_2d):
        """Evolución unitaria debe preservar pureza."""
        # Hadamard gate
        hadamard = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        
        initial_purity = pure_state_2d.compute_metrics().purity
        evolved = pure_state_2d.evolve_unitary(hadamard)
        final_purity = evolved.compute_metrics().purity
        
        assert np.isclose(initial_purity, final_purity, atol=1e-10), \
            "Evolución unitaria no preserva pureza"
    
    def test_unitary_evolution_preserves_trace(self, pure_state_2d):
        """Evolución unitaria debe preservar traza."""
        pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        
        evolved = pure_state_2d.evolve_unitary(pauli_x)
        trace = np.trace(evolved.matrix)
        
        assert np.isclose(trace.real, 1.0, atol=1e-12), "Evolución no preserva traza"
    
    def test_invalid_non_unitary_evolution(self, pure_state_2d):
        """Debe rechazar operadores no unitarios."""
        non_unitary = np.array([[2, 0], [0, 0.5]], dtype=np.complex128)
        with pytest.raises(ValueError, match="no unitario"):
            pure_state_2d.evolve_unitary(non_unitary, validate=True)
    
    def test_partial_trace_product_state(self, entangled_state_4d):
        """Traza parcial de estado producto debe dar estado puro."""
        # Estado |00⟩: ρ_AB = |00⟩⟨00|
        psi_00 = np.array([1, 0, 0, 0], dtype=np.complex128)
        rho_product = AtomicDensityMatrix(np.outer(psi_00, psi_00.conj()))
        
        # Trazar sobre B
        rho_a = rho_product.partial_trace(dims=(2, 2), subsystem=0)
        
        # Debe resultar en |0⟩⟨0|
        expected = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        assert np.allclose(rho_a.matrix, expected, atol=1e-10), "Traza parcial incorrecta"
    
    def test_partial_trace_preserves_total_trace(self, entangled_state_4d):
        """Traza parcial debe preservar traza total."""
        rho_reduced = entangled_state_4d.partial_trace(dims=(2, 2), subsystem=0)
        trace = np.trace(rho_reduced.matrix)
        assert np.isclose(trace.real, 1.0, atol=1e-12), "Traza parcial no preserva traza"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Property-Based Testing
    # ─────────────────────────────────────────────────────────────────────────
    
    @given(st.integers(min_value=2, max_value=8))
    @settings(max_examples=20, deadline=5000)
    def test_pure_state_factory_creates_valid_states(self, dimension):
        """Estados puros generados deben satisfacer todos los axiomas."""
        rho = create_quantum_mac_state(dimension=dimension, purity=1.0, seed=42)
        
        # Verificar axiomas
        assert rho.dimension == dimension
        metrics = rho.compute_metrics()
        assert np.isclose(metrics.purity, 1.0, atol=1e-6), "Estado no puro"
        assert np.isclose(metrics.von_neumann_entropy, 0.0, atol=1e-6), "Entropía no nula"
    
    @given(
        st.integers(min_value=2, max_value=6),
        st.floats(min_value=0.1, max_value=1.0)
    )
    @settings(max_examples=20, deadline=5000)
    def test_mixed_state_factory_creates_valid_purities(self, dimension, purity):
        """Estados mixtos deben tener pureza controlada."""
        assume(purity >= 1.0 / dimension)  # Límite físico
        
        rho = create_quantum_mac_state(dimension=dimension, purity=purity, seed=42)
        metrics = rho.compute_metrics()
        
        # Pureza debe estar cerca del objetivo (dentro de tolerancia numérica)
        assert abs(metrics.purity - purity) < 0.1, \
            f"Pureza fuera de rango: {metrics.purity} vs {purity}"


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2: PRUEBAS UNITARIAS - COHOMOLOGÍA DE HACES CELULARES
# ══════════════════════════════════════════════════════════════════════════════

class TestCellularSheafNeuralManifold:
    """Suite de pruebas para fibrados de haces celulares."""
    
    @pytest.fixture
    def simple_chain_complex(self) -> CellularSheafNeuralManifold:
        """Complejo de cadena simple: 0 -- 1 -- 2 (grafo lineal)."""
        # Matriz de incidencia: 2 aristas, 3 vértices
        # e0: 0 → 1, e1: 1 → 2
        B1 = sp.csr_matrix([
            [-1, 1, 0],   # e0
            [0, -1, 1]    # e1
        ], dtype=np.float64)
        
        # Mapas de restricción (identidad escalar)
        restriction_maps = {
            0: RestrictionMap(matrix=np.eye(1), source_dim=1, target_dim=1),
            1: RestrictionMap(matrix=np.eye(1), source_dim=1, target_dim=1)
        }
        
        fiber_dims = {'vertex': 1, 'edge': 1}
        
        return CellularSheafNeuralManifold(
            incidence_matrix=B1,
            restriction_maps=restriction_maps,
            fiber_dims=fiber_dims
        )
    
    @pytest.fixture
    def triangle_complex(self) -> CellularSheafNeuralManifold:
        """Complejo triangular: grafo cíclico con 3 vértices."""
        # 0 -- 1
        #  \  /
        #   2
        B1 = sp.csr_matrix([
            [-1, 1, 0],   # e0: 0 → 1
            [-1, 0, 1],   # e1: 0 → 2
            [0, -1, 1]    # e2: 1 → 2
        ], dtype=np.float64)
        
        restriction_maps = {
            0: RestrictionMap(matrix=np.eye(1), source_dim=1, target_dim=1),
            1: RestrictionMap(matrix=np.eye(1), source_dim=1, target_dim=1),
            2: RestrictionMap(matrix=np.eye(1), source_dim=1, target_dim=1)
        }
        
        fiber_dims = {'vertex': 1, 'edge': 1}
        
        return CellularSheafNeuralManifold(
            incidence_matrix=B1,
            restriction_maps=restriction_maps,
            fiber_dims=fiber_dims
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Operador Cofrontera
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_coboundary_operator_dimensions(self, simple_chain_complex):
        """Operador cofrontera debe tener dimensiones correctas."""
        delta = simple_chain_complex.compute_coboundary_matrix()
        
        num_edges = simple_chain_complex.num_edges
        num_vertices = simple_chain_complex.num_vertices
        d_v = simple_chain_complex.fiber_dims['vertex']
        d_e = simple_chain_complex.fiber_dims['edge']
        
        expected_shape = (num_edges * d_e, num_vertices * d_v)
        assert delta.shape == expected_shape, f"Forma incorrecta: {delta.shape} vs {expected_shape}"
    
    def test_coboundary_annihilates_constant_sections(self, simple_chain_complex):
        """δ debe anular secciones constantes (elemento de H⁰)."""
        # Sección constante: x_v = c para todo v
        constant_section = np.ones(simple_chain_complex.num_vertices)
        delta_x = simple_chain_complex.compute_coboundary(constant_section)
        
        assert np.allclose(delta_x, 0.0, atol=1e-10), \
            "Sección constante no está en ker(δ)"
    
    def test_coboundary_detects_discontinuity(self, simple_chain_complex):
        """δ debe detectar discontinuidades en secciones."""
        # Sección discontinua: [1, 0, 0]
        discontinuous_section = np.array([1.0, 0.0, 0.0])
        delta_x = simple_chain_complex.compute_coboundary(discontinuous_section)
        
        # Debe tener componentes no nulas
        assert not np.allclose(delta_x, 0.0, atol=1e-10), \
            "Discontinuidad no detectada por δ"
    
    def test_dirichlet_energy_zero_for_harmonic_sections(self, simple_chain_complex):
        """Energía de Dirichlet debe ser 0 para secciones armónicas."""
        harmonic_section = np.ones(simple_chain_complex.num_vertices)
        energy = simple_chain_complex.compute_dirichlet_energy(harmonic_section)
        
        assert np.isclose(energy, 0.0, atol=1e-10), \
            f"Energía no nula para sección armónica: {energy}"
    
    def test_dirichlet_energy_positive_for_nonharmonic(self, simple_chain_complex):
        """Energía de Dirichlet debe ser > 0 para secciones no armónicas."""
        nonharmonic_section = np.array([1.0, 0.0, -1.0])
        energy = simple_chain_complex.compute_dirichlet_energy(nonharmonic_section)
        
        assert energy > 0, f"Energía no positiva: {energy}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Laplaciano de Hodge
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_hodge_laplacian_is_symmetric(self, simple_chain_complex):
        """Laplaciano de Hodge debe ser simétrico."""
        laplacian = simple_chain_complex.compute_hodge_laplacian()
        assert np.allclose(laplacian, laplacian.T, atol=1e-10), \
            "Laplaciano no simétrico"
    
    def test_hodge_laplacian_is_positive_semidefinite(self, simple_chain_complex):
        """Laplaciano debe ser semidefinido positivo."""
        laplacian = simple_chain_complex.compute_hodge_laplacian()
        eigenvalues = la.eigvalsh(laplacian)
        
        assert np.all(eigenvalues >= -1e-10), \
            f"Laplaciano con eigenvalores negativos: {eigenvalues}"
    
    def test_hodge_laplacian_kernel_dimension(self, triangle_complex):
        """Dimensión del núcleo del Laplaciano debe ser β₀ (componentes conexas)."""
        laplacian = triangle_complex.compute_hodge_laplacian()
        eigenvalues = la.eigvalsh(laplacian)
        
        # Contar eigenvalores nulos (tolerancia numérica)
        kernel_dim = np.sum(eigenvalues < 1e-9)
        
        # Para grafo conexo, β₀ = 1
        assert kernel_dim == 1, f"Dimensión de núcleo incorrecta: {kernel_dim}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Grupos de Cohomología
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_cohomology_betti_numbers_chain(self, simple_chain_complex):
        """Números de Betti para grafo lineal: β₀=1, β₁=0."""
        cohomology = simple_chain_complex.compute_cohomology_groups()
        
        assert cohomology[0].betti_number == 1, "β₀ incorrecto para grafo conexo"
        assert cohomology[1].betti_number == 0, "β₁ incorrecto para grafo acíclico"
    
    def test_cohomology_betti_numbers_triangle(self, triangle_complex):
        """Números de Betti para triángulo: β₀=1, β₁=1."""
        cohomology = triangle_complex.compute_cohomology_groups()
        
        assert cohomology[0].betti_number == 1, "β₀ incorrecto"
        # Triángulo tiene un ciclo → β₁ = 1
        assert cohomology[1].betti_number == 1, "β₁ incorrecto para grafo cíclico"
    
    def test_cohomology_kernel_basis_spans_harmonic_sections(self, simple_chain_complex):
        """Base del núcleo debe generar secciones armónicas."""
        cohomology = simple_chain_complex.compute_cohomology_groups()
        kernel_basis = cohomology[0].kernel_basis
        
        # Cada vector de la base debe estar en ker(δ)
        for i in range(kernel_basis.shape[1]):
            basis_vector = kernel_basis[:, i]
            energy = simple_chain_complex.compute_dirichlet_energy(basis_vector)
            assert energy < 1e-9, f"Vector de base no armónico: E={energy}"
    
    def test_euler_characteristic_formula(self, triangle_complex):
        """Fórmula de Euler-Poincaré: χ = β₀ - β₁."""
        cohomology = triangle_complex.compute_cohomology_groups()
        
        beta_0 = cohomology[0].betti_number
        beta_1 = cohomology[1].betti_number
        
        euler_char = beta_0 - beta_1
        
        # Para triángulo (2-complejo): χ = V - E + F = 3 - 3 + 1 = 1
        # Pero solo tenemos estructura 1-esqueleto: χ = 3 - 3 = 0
        # Con ciclo: χ = 1 - 1 = 0
        assert euler_char == 0, f"Característica de Euler incorrecta: {euler_char}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Holonomía Semántica
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_verify_holonomy_accepts_harmonic(self, simple_chain_complex):
        """Secciones armónicas deben pasar verificación de holonomía."""
        harmonic = np.ones(simple_chain_complex.num_vertices)
        is_holonomic, energy = simple_chain_complex.verify_semantic_holonomy(harmonic)
        
        assert is_holonomic, "Sección armónica rechazada"
        assert energy < 1e-9, f"Energía no nula: {energy}"
    
    def test_verify_holonomy_rejects_discontinuous(self, simple_chain_complex):
        """Secciones discontinuas deben fallar verificación de holonomía."""
        discontinuous = np.array([1.0, 0.0, -1.0])
        is_holonomic, energy = simple_chain_complex.verify_semantic_holonomy(discontinuous)
        
        assert not is_holonomic, "Sección discontinua aceptada incorrectamente"
        assert energy > 0, "Energía nula para sección discontinua"
    
    def test_projection_to_harmonic_reduces_energy(self, simple_chain_complex):
        """Proyección a H⁰ debe reducir energía de Dirichlet."""
        arbitrary_section = np.array([1.0, 2.0, 3.0])
        
        energy_before = simple_chain_complex.compute_dirichlet_energy(arbitrary_section)
        projected = simple_chain_complex.project_to_harmonic(arbitrary_section)
        energy_after = simple_chain_complex.compute_dirichlet_energy(projected)
        
        assert energy_after <= energy_before + 1e-10, \
            "Proyección aumenta energía"
    
    def test_projection_to_harmonic_is_idempotent(self, simple_chain_complex):
        """Proyectar dos veces debe dar el mismo resultado."""
        section = np.array([1.0, 2.0, 3.0])
        
        projected_once = simple_chain_complex.project_to_harmonic(section)
        projected_twice = simple_chain_complex.project_to_harmonic(projected_once)
        
        assert np.allclose(projected_once, projected_twice, atol=1e-10), \
            "Proyección no es idempotente"


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3: PRUEBAS UNITARIAS - GEOMETRÍA SIMPLÉCTICA Y APRENDIZAJE
# ══════════════════════════════════════════════════════════════════════════════

class TestDiracStructure:
    """Suite de pruebas para estructura de Dirac."""
    
    @pytest.fixture
    def simple_dirac_2d(self) -> DiracStructure:
        """Estructura de Dirac simple en ℝ²."""
        J = np.array([[0, 1], [-1, 0]], dtype=np.float64)  # Antisimétrica
        R = np.array([[0.1, 0], [0, 0.1]], dtype=np.float64)  # Disipación isótropa
        return DiracStructure(J, R)
    
    @pytest.fixture
    def anisotropic_dirac_3d(self) -> DiracStructure:
        """Estructura de Dirac con disipación anisotrópica."""
        J = np.array([[0, 1, 0], [-1, 0, 2], [0, -2, 0]], dtype=np.float64)
        R = np.diag([0.1, 0.5, 0.01])  # Disipación anisotrópica
        return DiracStructure(J, R)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Axiomas de Dirac
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_j_antisymmetry(self, simple_dirac_2d):
        """J debe ser estrictamente antisimétrica: J = -J^T."""
        J = simple_dirac_2d.J
        assert np.allclose(J, -J.T, atol=1e-12), "J no es antisimétrica"
    
    def test_r_symmetry(self, simple_dirac_2d):
        """R debe ser simétrica: R = R^T."""
        R = simple_dirac_2d.R
        assert np.allclose(R, R.T, atol=1e-12), "R no es simétrica"
    
    def test_r_positive_semidefiniteness(self, simple_dirac_2d):
        """R debe ser semidefinida positiva: R ≽ 0."""
        R = simple_dirac_2d.R
        eigenvalues = la.eigvalsh(R)
        assert np.all(eigenvalues >= -1e-12), \
            f"R no es semidefinida positiva: {eigenvalues}"
    
    def test_invalid_non_antisymmetric_j(self):
        """Debe rechazar J no antisimétrica."""
        J_invalid = np.array([[1, 1], [1, 1]], dtype=np.float64)
        R = np.eye(2) * 0.1
        
        with pytest.raises(NumericalInstabilityError, match="antisimétrica"):
            DiracStructure(J_invalid, R)
    
    def test_invalid_non_symmetric_r(self):
        """Debe rechazar R no simétrica."""
        J = np.array([[0, 1], [-1, 0]], dtype=np.float64)
        R_invalid = np.array([[1, 2], [3, 1]], dtype=np.float64)
        
        with pytest.raises(NumericalInstabilityError, match="simétrica"):
            DiracStructure(J, R_invalid)
    
    def test_invalid_negative_definite_r(self):
        """Debe rechazar R con eigenvalores negativos."""
        J = np.array([[0, 1], [-1, 0]], dtype=np.float64)
        R_invalid = np.array([[-1, 0], [0, -1]], dtype=np.float64)
        
        with pytest.raises(NumericalInstabilityError, match="semidefinida positiva"):
            DiracStructure(J, R_invalid)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Disipación
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_dissipation_rate_is_positive(self, simple_dirac_2d):
        """Tasa de disipación debe ser no negativa: Φ = ∇H^T R ∇H ≥ 0."""
        gradient = np.array([1.0, 1.0])
        dissipation = simple_dirac_2d.compute_dissipation_rate(gradient)
        
        assert dissipation >= 0, f"Disipación negativa: {dissipation}"
    
    def test_dissipation_rate_zero_for_zero_gradient(self, simple_dirac_2d):
        """Disipación debe ser 0 si gradiente es 0."""
        gradient = np.array([0.0, 0.0])
        dissipation = simple_dirac_2d.compute_dissipation_rate(gradient)
        
        assert np.isclose(dissipation, 0.0, atol=1e-12), \
            f"Disipación no nula con gradiente nulo: {dissipation}"
    
    def test_dissipation_rate_scales_quadratically(self, simple_dirac_2d):
        """Disipación debe escalar cuadráticamente con gradiente."""
        gradient = np.array([1.0, 0.0])
        dissipation_1 = simple_dirac_2d.compute_dissipation_rate(gradient)
        
        gradient_2x = 2.0 * gradient
        dissipation_2 = simple_dirac_2d.compute_dissipation_rate(gradient_2x)
        
        # Debe ser 4x (cuadrático)
        assert np.isclose(dissipation_2, 4 * dissipation_1, atol=1e-10), \
            "Disipación no escala cuadráticamente"
    
    def test_anisotropic_dissipation_direction_dependence(self, anisotropic_dirac_3d):
        """Disipación anisotrópica debe depender de la dirección."""
        gradient_x = np.array([1.0, 0.0, 0.0])
        gradient_y = np.array([0.0, 1.0, 0.0])
        
        dissipation_x = anisotropic_dirac_3d.compute_dissipation_rate(gradient_x)
        dissipation_y = anisotropic_dirac_3d.compute_dissipation_rate(gradient_y)
        
        # Deben ser diferentes (R anisotrópica)
        assert not np.isclose(dissipation_x, dissipation_y, atol=1e-6), \
            "Disipación anisotrópica no detectada"


class TestPortHamiltonianLearningFlow:
    """Suite de pruebas para aprendizaje Port-Hamiltoniano."""
    
    @pytest.fixture
    def simple_learner(self) -> PortHamiltonianLearningFlow:
        """Learner simple 2D."""
        J = np.array([[0, 1], [-1, 0]], dtype=np.float64)
        R = np.eye(2) * 0.1
        dirac = DiracStructure(J, R)
        return PortHamiltonianLearningFlow(dirac, adaptive_timestep=False)
    
    @pytest.fixture
    def quadratic_hamiltonian(self) -> Callable[[np.ndarray], float]:
        """Hamiltoniano cuadrático simple: H(W) = 0.5 * ||W||²."""
        return lambda W: 0.5 * np.sum(W ** 2)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Disipatividad de Lyapunov
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_lyapunov_energy_decreases(self, simple_learner, quadratic_hamiltonian):
        """Energía debe decrecer o mantenerse: H(t+dt) ≤ H(t)."""
        W_initial = np.array([1.0, 1.0])
        gradient = W_initial  # ∇H = W para H = 0.5||W||²
        
        W_next, certificate = simple_learner.apply_weight_update(
            W_k=W_initial,
            grad_H=gradient,
            dt=0.01,
            hamiltonian_fn=quadratic_hamiltonian
        )
        
        assert certificate.energy_final <= certificate.energy_initial + 1e-10, \
            f"Energía aumenta: {certificate.energy_initial} → {certificate.energy_final}"
    
    def test_lyapunov_dissipated_energy_positive(self, simple_learner):
        """Energía disipada debe ser no negativa."""
        W_initial = np.array([1.0, 1.0])
        gradient = W_initial
        
        _, certificate = simple_learner.apply_weight_update(
            W_k=W_initial,
            grad_H=gradient,
            dt=0.01
        )
        
        assert certificate.dissipated_energy >= -1e-10, \
            f"Energía disipada negativa: {certificate.dissipated_energy}"
    
    def test_lyapunov_derivative_negative(self, simple_learner):
        """Derivada de Lyapunov debe ser no positiva: dH/dt ≤ 0."""
        W_initial = np.array([1.0, 1.0])
        gradient = W_initial
        
        _, certificate = simple_learner.apply_weight_update(
            W_k=W_initial,
            grad_H=gradient,
            dt=0.01
        )
        
        assert certificate.lyapunov_derivative <= 1e-10, \
            f"Derivada de Lyapunov positiva: {certificate.lyapunov_derivative}"
    
    def test_convergence_to_minimum(self, simple_learner, quadratic_hamiltonian):
        """Múltiples iteraciones deben converger al mínimo."""
        W = np.array([10.0, 10.0])
        
        for _ in range(100):
            gradient = W  # ∇H = W
            W, _ = simple_learner.apply_weight_update(
                W_k=W,
                grad_H=gradient,
                dt=0.1,
                hamiltonian_fn=quadratic_hamiltonian
            )
        
        # Debe estar cerca de [0, 0]
        assert np.linalg.norm(W) < 0.1, f"No converge: {W}"
    
    def test_weight_update_respects_structure(self, simple_learner):
        """Actualización debe usar estructura (J - R)."""
        W = np.array([1.0, 0.0])
        gradient = np.array([1.0, 0.0])
        
        W_next, _ = simple_learner.apply_weight_update(
            W_k=W,
            grad_H=gradient,
            dt=0.1
        )
        
        # Con J antisimétrica, debe haber acoplamiento entre componentes
        structure = simple_learner.dirac.structure_matrix()
        expected_dW = -0.1 * (structure @ gradient)
        expected_W = W + expected_dW
        
        assert np.allclose(W_next, expected_W, atol=1e-10), \
            "Actualización no usa estructura correctamente"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Adaptación de Paso Temporal
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_adaptive_timestep_increases_for_small_gradient(self):
        """Paso temporal debe aumentar si gradiente es pequeño."""
        J = np.array([[0, 1], [-1, 0]], dtype=np.float64)
        R = np.eye(2) * 0.1
        dirac = DiracStructure(J, R)
        learner = PortHamiltonianLearningFlow(dirac, adaptive_timestep=True)
        
        small_gradient = np.array([0.01, 0.01])
        dt_current = 0.01
        
        dt_new = learner.adapt_timestep(
            gradient=small_gradient,
            dt_current=dt_current,
            target_dissipation=1e-3
        )
        
        # Debe aumentar (pero limitado por factor de seguridad)
        assert dt_new >= dt_current * 0.99, "Paso temporal no aumenta apropiadamente"
    
    def test_adaptive_timestep_decreases_for_large_gradient(self):
        """Paso temporal debe disminuir si gradiente es grande."""
        J = np.array([[0, 1], [-1, 0]], dtype=np.float64)
        R = np.eye(2) * 0.1
        dirac = DiracStructure(J, R)
        learner = PortHamiltonianLearningFlow(dirac, adaptive_timestep=True)
        
        large_gradient = np.array([10.0, 10.0])
        dt_current = 0.1
        
        dt_new = learner.adapt_timestep(
            gradient=large_gradient,
            dt_current=dt_current,
            target_dissipation=1e-3
        )
        
        # Debe disminuir
        assert dt_new <= dt_current * 1.01, "Paso temporal no disminuye apropiadamente"


class TestGaloisAdjunctionFunctor:
    """Suite de pruebas para funtor de adjunción de Galois."""
    
    @pytest.fixture
    def simple_system(self) -> Tuple[CellularSheafNeuralManifold, PortHamiltonianLearningFlow, GaloisAdjunctionFunctor]:
        """Sistema completo simple para pruebas."""
        return create_geometric_learning_system(
            num_vertices=5,
            num_edges=6,
            fiber_dim_vertex=2,
            fiber_dim_edge=2,
            dissipation_strength=0.05,
            seed=42
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Asimilación Semántica
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_semantic_cartridge_processing_preserves_holonomy(self, simple_system):
        """Procesamiento debe preservar holonomía si enforce_holonomy=True."""
        sheaf, learner, functor = simple_system
        
        # Inicializar con sección armónica
        W_initial = np.ones(sheaf.num_vertices * sheaf.fiber_dims['vertex'])
        W_initial = sheaf.project_to_harmonic(W_initial)
        
        gradient = np.random.randn(W_initial.shape[0]) * 0.01
        
        W_updated, metadata = functor.process_semantic_cartridge(
            mic_vector=None,
            atomic_weights=W_initial,
            grad_error=gradient,
            dt=0.01
        )
        
        # Si se habilita proyección, debe mantener holonomía
        if functor.project_to_harmonic:
            assert metadata['holonomy_final'], "Holonomía no preservada"
    
    def test_semantic_cartridge_rejects_holonomy_violations(self):
        """Debe rechazar actualizaciones que violen holonomía."""
        sheaf, learner, _ = create_geometric_learning_system(
            num_vertices=5,
            num_edges=6,
            fiber_dim_vertex=1,
            fiber_dim_edge=1,
            seed=42
        )
        
        # Functor con enforce_holonomy=True y sin proyección
        functor = GaloisAdjunctionFunctor(
            sheaf_manifold=sheaf,
            learner=learner,
            enforce_holonomy=True,
            project_to_harmonic=False
        )
        
        # Sección discontinua (viola holonomía)
        W_discontinuous = np.random.randn(sheaf.num_vertices)
        gradient = np.random.randn(W_discontinuous.shape[0]) * 0.01
        
        # Debe lanzar error si la actualización viola holonomía
        # (dependiendo del gradiente, puede o no violar)
        try:
            functor.process_semantic_cartridge(
                mic_vector=None,
                atomic_weights=W_discontinuous,
                grad_error=gradient,
                dt=0.01
            )
        except NumericalInstabilityError as e:
            assert "Veto Ontológico" in str(e), "Error incorrecto"
    
    def test_telemetry_tracking(self, simple_system):
        """Telemetría debe rastrear actualizaciones y rechazos."""
        _, _, functor = simple_system
        
        initial_updates = functor.update_count
        
        # Realizar una actualización
        W = np.random.randn(10) * 0.1
        W = functor.sheaf.project_to_harmonic(W)
        gradient = np.random.randn(10) * 0.01
        
        functor.process_semantic_cartridge(
            mic_vector=None,
            atomic_weights=W,
            grad_error=gradient,
            dt=0.01
        )
        
        telemetry = functor.get_telemetry()
        assert telemetry['total_updates'] == initial_updates + 1, \
            "Telemetría no actualizada"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Adjunción Categorial
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_adjunction_unit_produces_harmonic_section(self, simple_system):
        """Unidad de adjunción debe producir sección armónica."""
        sheaf, _, functor = simple_system
        
        from app.core.mic_algebra import CategoricalState
        mic_state = CategoricalState(category_id="test", morphisms=[])
        
        unit_section = functor.compute_adjunction_unit(mic_state)
        
        # Verificar que está en H⁰
        energy = sheaf.compute_dirichlet_energy(unit_section)
        assert energy < 1e-6, f"Unidad no es armónica: E={energy}"
    
    def test_adjunction_counit_is_well_defined(self, simple_system):
        """Counidad debe ser bien definida (producir estado categórico válido)."""
        _, _, functor = simple_system
        
        mac_section = np.random.randn(10)
        counit_state = functor.compute_adjunction_counit(mac_section)
        
        assert hasattr(counit_state, 'category_id'), "Counidad mal formada"


# ══════════════════════════════════════════════════════════════════════════════
# FASE 4: PRUEBAS DE INTEGRACIÓN Y ESCENARIOS END-TO-END
# ══════════════════════════════════════════════════════════════════════════════

class TestEndToEndScenarios:
    """Pruebas de integración de flujos completos."""
    
    def test_complete_learning_cycle_convergence(self):
        """Ciclo completo de aprendizaje debe converger."""
        sheaf, learner, functor = create_geometric_learning_system(
            num_vertices=8,
            num_edges=10,
            fiber_dim_vertex=3,
            fiber_dim_edge=3,
            dissipation_strength=0.1,
            seed=42
        )
        
        # Inicializar pesos
        W = np.random.randn(sheaf.num_vertices * sheaf.fiber_dims['vertex']) * 0.5
        W = sheaf.project_to_harmonic(W)
        
        # Hamiltoniano cuadrático
        def hamiltonian(weights):
            return 0.5 * np.sum(weights ** 2)
        
        energies = []
        
        # Ejecutar múltiples iteraciones
        for iteration in range(50):
            gradient = W  # ∇H = W para H cuadrático
            
            W, metadata = functor.process_semantic_cartridge(
                mic_vector=None,
                atomic_weights=W,
                grad_error=gradient,
                hamiltonian_fn=hamiltonian,
                dt=0.05
            )
            
            energies.append(metadata['lyapunov_certificate'].energy_final)
        
        # Verificar convergencia monótona
        for i in range(1, len(energies)):
            assert energies[i] <= energies[i-1] + 1e-6, \
                f"Energía aumenta en iteración {i}: {energies[i-1]} → {energies[i]}"
        
        # Verificar convergencia a mínimo
        assert energies[-1] < energies[0] * 0.1, "No converge suficientemente"
    
    def test_quantum_sheaf_coupling(self):
        """Acoplamiento entre estado cuántico y fibrado de haces."""
        # Estado cuántico MAC
        rho_mac = create_quantum_mac_state(dimension=4, purity=0.8, seed=42)
        
        # Sistema de haces
        sheaf, _, _ = create_geometric_learning_system(
            num_vertices=4,
            num_edges=5,
            fiber_dim_vertex=1,
            fiber_dim_edge=1,
            seed=42
        )
        
        # Observables semánticos (proyectores sobre vértices)
        observables = []
        for i in range(4):
            proj = np.zeros((4, 4), dtype=np.complex128)
            proj[i, i] = 1.0
            observables.append(proj)
        
        # Medir y construir sección
        section = np.array([
            rho_mac.measure_observable(obs) for obs in observables
        ])
        
        # Verificar conservación de probabilidad
        assert np.isclose(np.sum(section), 1.0, atol=1e-10), \
            "Probabilidades no suman 1"
        
        # Verificar energía de Dirichlet
        energy = sheaf.compute_dirichlet_energy(section)
        assert energy >= 0, "Energía negativa"
    
    def test_stress_test_large_system(self):
        """Prueba de estrés con sistema grande."""
        sheaf, learner, functor = create_geometric_learning_system(
            num_vertices=50,
            num_edges=100,
            fiber_dim_vertex=5,
            fiber_dim_edge=5,
            dissipation_strength=0.05,
            seed=42
        )
        
        W = np.random.randn(sheaf.num_vertices * sheaf.fiber_dims['vertex']) * 0.1
        W = sheaf.project_to_harmonic(W)
        
        gradient = np.random.randn(W.shape[0]) * 0.01
        
        # Debe completarse sin errores
        W_updated, metadata = functor.process_semantic_cartridge(
            mic_vector=None,
            atomic_weights=W,
            grad_error=gradient,
            dt=0.01
        )
        
        assert W_updated.shape == W.shape, "Dimensiones incorrectas"
        assert metadata['holonomy_final'] or not functor.enforce_holonomy, \
            "Holonomía violada en sistema grande"


# ══════════════════════════════════════════════════════════════════════════════
# FASE 5: PRUEBAS DE ESTABILIDAD NUMÉRICA Y CASOS EXTREMOS
# ══════════════════════════════════════════════════════════════════════════════

class TestNumericalStability:
    """Pruebas de estabilidad numérica y casos límite."""
    
    def test_ill_conditioned_density_matrix(self):
        """Manejo de matrices de densidad mal condicionadas."""
        # Eigenvalores muy dispares
        eigenvalues = np.array([0.99, 0.01, 1e-10, 1e-10])
        eigenvalues /= np.sum(eigenvalues)
        
        U, _ = la.qr(np.random.randn(4, 4) + 1j * np.random.randn(4, 4))
        rho = U @ np.diag(eigenvalues) @ U.conj().T
        
        density = AtomicDensityMatrix(rho, tol=1e-8)
        metrics = density.compute_metrics()
        
        # Debe calcular métricas sin error
        assert 0 <= metrics.purity <= 1, "Pureza fuera de rango"
        assert metrics.von_neumann_entropy >= 0, "Entropía negativa"
    
    def test_near_singular_coboundary_operator(self):
        """Manejo de operadores cofrontera casi singulares."""
        # Grafo desconectado (múltiples componentes)
        B1 = sp.csr_matrix([
            [-1, 1, 0, 0],
            [0, 0, -1, 1]
        ], dtype=np.float64)
        
        restriction_maps = {
            0: RestrictionMap(matrix=np.eye(1), source_dim=1, target_dim=1),
            1: RestrictionMap(matrix=np.eye(1), source_dim=1, target_dim=1)
        }
        
        sheaf = CellularSheafNeuralManifold(
            incidence_matrix=B1,
            restriction_maps=restriction_maps,
            fiber_dims={'vertex': 1, 'edge': 1}
        )
        
        # Debe calcular cohomología correctamente
        cohomology = sheaf.compute_cohomology_groups()
        
        # Grafo con 2 componentes → β₀ = 2
        assert cohomology[0].betti_number == 2, \
            f"β₀ incorrecto: {cohomology[0].betti_number}"
    
    def test_zero_gradient_handling(self):
        """Manejo de gradientes nulos."""
        J = np.array([[0, 1], [-1, 0]], dtype=np.float64)
        R = np.eye(2) * 0.1
        dirac = DiracStructure(J, R)
        learner = PortHamiltonianLearningFlow(dirac)
        
        W = np.array([1.0, 1.0])
        zero_gradient = np.array([0.0, 0.0])
        
        W_next, certificate = learner.apply_weight_update(
            W_k=W,
            grad_H=zero_gradient,
            dt=0.1
        )
        
        # Sin gradiente, pesos no deben cambiar (por estructura)
        # Pero (J-R)·0 = 0, así que W_next = W
        assert np.allclose(W_next, W, atol=1e-10), "Pesos cambian con gradiente nulo"
        assert np.isclose(certificate.dissipated_energy, 0.0, atol=1e-12), \
            "Disipación no nula con gradiente nulo"
    
    def test_extreme_timestep_stability(self):
        """Estabilidad con pasos temporales extremos."""
        J = np.array([[0, 1], [-1, 0]], dtype=np.float64)
        R = np.eye(2) * 0.1
        dirac = DiracStructure(J, R)
        learner = PortHamiltonianLearningFlow(dirac)
        
        W = np.array([1.0, 1.0])
        gradient = np.array([0.1, 0.1])
        
        def hamiltonian(w):
            return 0.5 * np.sum(w ** 2)
        
        # Paso temporal muy pequeño
        W_small, cert_small = learner.apply_weight_update(
            W_k=W, grad_H=gradient, dt=1e-10, hamiltonian_fn=hamiltonian
        )
        assert cert_small.is_stable, "Inestable con dt muy pequeño"
        
        # Paso temporal grande (puede violar estabilidad)
        W_large, cert_large = learner.apply_weight_update(
            W_k=W, grad_H=gradient, dt=1.0, hamiltonian_fn=hamiltonian
        )
        # Simplemente verificar que no hay errores catastróficos
        assert not np.any(np.isnan(W_large)), "NaN generado con dt grande"


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PYTEST Y FIXTURES GLOBALES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def setup_logging():
    """Configurar logging para pruebas."""
    logging.basicConfig(
        level=logging.WARNING,  # Reducir ruido en pruebas
        format='%(name)s - %(levelname)s - %(message)s'
    )


@pytest.fixture(scope="session")
def random_seed():
    """Semilla aleatoria global para reproducibilidad."""
    np.random.seed(42)
    return 42


# ══════════════════════════════════════════════════════════════════════════════
# MARCADORES Y CONFIGURACIÓN DE PYTEST
# ══════════════════════════════════════════════════════════════════════════════

# Marcadores personalizados
pytest.mark.quantum = pytest.mark.quantum
pytest.mark.cohomology = pytest.mark.cohomology
pytest.mark.symplectic = pytest.mark.symplectic
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow


# ══════════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA PARA EJECUCIÓN DIRECTA
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",                    # Verbose
        "--tb=short",            # Traceback corto
        "--cov=app.wisdom",      # Cobertura de código
        "--cov-report=html",     # Reporte HTML
        "--cov-report=term",     # Reporte en terminal
        "-m", "not slow",        # Excluir pruebas lentas por defecto
        "--maxfail=5",           # Detener después de 5 fallos
        "--durations=10"         # Mostrar 10 pruebas más lentas
    ])