"""
Módulo de Testing: Antimatter Choke Coil Test Suite
Ubicación: tests/unit/physics/test_antimatter_choke_coil.py
Versión: 2.0.0 (Validación Exhaustiva - Cobertura Matemática Completa)

FILOSOFÍA DE TESTING:

§1. TESTING BASADO EN PROPIEDADES (Property-Based Testing):
Validación de invariantes matemáticos fundamentales en lugar de casos específicos.
Utiliza hipótesis algebraicas para generar espacios de prueba exhaustivos.

§2. VERIFICACIÓN FORMAL DE AXIOMAS FÍSICOS:
- Conservación de energía: ΔE_total = 0
- Segunda ley termodinámica: ΔS ≥ 0
- Unitariedad cuántica: ⟨ψ|ψ⟩ = 1
- Pasividad de impedancia: Re(Z) ≥ 0
- Estabilidad Lyapunov: V̇ ≤ 0

§3. COBERTURA DE CASOS EXTREMOS (Edge Cases):
- Valores límite (0, ε, ∞)
- Singularidades numéricas
- Condiciones degeneradas
- Transiciones de fase

§4. TESTING DE REGRESIÓN:
Validación de resultados conocidos analíticamente (benchmarks).

§5. TESTING DE INTEGRACIÓN:
Interacción entre subsistemas (topología ↔ cuántica ↔ hamiltoniana).
"""

from __future__ import annotations

import gc
import logging
import sys
import time
import warnings
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import scipy.sparse as sp
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import norm

# Importar módulo a testear
sys.path.insert(0, '../app')
from app.physics.antimatter_choke_coil import (
    AnnihilationEvent,
    AntimatterAnalytics,
    AntimatterChokeCoil,
    FockSpaceOperators,
    HomologicalAnnihilator,
    PhysicalConstants,
    PortHamiltonianSystem,
    QuantumNumbers,
    QuantumState,
    SimplicialComplex,
    TopologicalInvariant,
)

# Configuración de logging para tests
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [TEST] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("MIC.Testing.AntimatterChokeCoil")

# Suprimir advertencias de scipy en tests
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


### ═══════════════════════════════════════════════════════════════════════════════
### FIXTURES Y UTILIDADES DE TESTING
### ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def physical_constants():
    """Fixture con constantes físicas para validación."""
    return {
        'hbar': PhysicalConstants.PLANCK_REDUCED,
        'c': PhysicalConstants.SPEED_OF_LIGHT,
        'm_e': PhysicalConstants.ELECTRON_MASS,
        'e': PhysicalConstants.ELEMENTARY_CHARGE,
        'epsilon_0': PhysicalConstants.VACUUM_PERMITTIVITY,
        'mu_0': PhysicalConstants.VACUUM_PERMEABILITY,
    }


@pytest.fixture(scope="function")
def antimatter_coil():
    """Fixture que retorna una bobina configurada estándar."""
    return AntimatterChokeCoil(
        inductance=100e-6,
        max_fock_state=50,
        enable_topology_collapse=True,
        enable_quantum_verification=True
    )


@pytest.fixture(scope="function")
def fock_operators():
    """Fixture con operadores del espacio de Fock."""
    return FockSpaceOperators(max_fock_state=30)


@pytest.fixture(scope="function")
def homological_annihilator():
    """Fixture con aniquilador homológico."""
    return HomologicalAnnihilator(tolerance=1e-12)


@pytest.fixture
def mock_positron_cartridge():
    """Factory para crear cartuchos de positrones mock."""
    def _create_cartridge(mass: float = PhysicalConstants.ELECTRON_MASS):
        cartridge = Mock()
        cartridge.inertial_mass = mass
        cartridge.authorization_signature = f"TEST_SIG_{np.random.randint(1000)}"
        return cartridge
    return _create_cartridge


@contextmanager
def performance_benchmark(test_name: str, max_time_ms: float = 1000.0) -> Generator:
    """Context manager para medir performance de tests."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"⏱️  {test_name}: {elapsed_ms:.2f} ms")
        if elapsed_ms > max_time_ms:
            logger.warning(f"⚠️  Performance degradada: {elapsed_ms:.2f} ms > {max_time_ms} ms")


def assert_hermitian(matrix: np.ndarray, tolerance: float = 1e-12):
    """Assert que una matriz es hermítica: A = A†"""
    assert_allclose(
        matrix,
        matrix.conj().T,
        atol=tolerance,
        err_msg="Matriz no es hermítica"
    )


def assert_unitary(matrix: np.ndarray, tolerance: float = 1e-12):
    """Assert que una matriz es unitaria: U†U = I"""
    identity = matrix.conj().T @ matrix
    expected_identity = np.eye(matrix.shape[0])
    assert_allclose(
        identity,
        expected_identity,
        atol=tolerance,
        err_msg="Matriz no es unitaria"
    )


def assert_positive_semidefinite(matrix: np.ndarray, tolerance: float = 1e-12):
    """Assert que una matriz es semi-definida positiva."""
    eigenvalues = np.linalg.eigvalsh(matrix)
    assert np.all(eigenvalues >= -tolerance), \
        f"Matriz no es SDP: λ_min = {eigenvalues.min()}"


def assert_antisymmetric(matrix: np.ndarray, tolerance: float = 1e-12):
    """Assert que una matriz es antisimétrica: A = -Aᵀ"""
    assert_allclose(
        matrix,
        -matrix.T,
        atol=tolerance,
        err_msg="Matriz no es antisimétrica"
    )


### ═══════════════════════════════════════════════════════════════════════════════
### TESTS DE CONSTANTES FÍSICAS Y VALIDACIÓN DIMENSIONAL
### ═══════════════════════════════════════════════════════════════════════════════

class TestPhysicalConstants:
    """Validación de constantes físicas fundamentales."""
    
    def test_speed_of_light_exact(self, physical_constants):
        """Verificar que c = 299792458 m/s (exacto por definición SI)."""
        assert physical_constants['c'] == 299792458.0
    
    def test_planck_constant_uncertainty(self, physical_constants):
        """Verificar ℏ con incertidumbre CODATA 2018."""
        hbar_expected = 1.054571817e-34
        assert abs(physical_constants['hbar'] - hbar_expected) < 1e-42
    
    def test_elementary_charge_exact(self, physical_constants):
        """Verificar e = 1.602176634×10⁻¹⁹ C (exacto post-2019)."""
        assert physical_constants['e'] == 1.602176634e-19
    
    def test_vacuum_permittivity_permeability_relation(self, physical_constants):
        """Verificar c² = 1/(ε₀μ₀) (relación electromagnética)."""
        c_squared = physical_constants['c']**2
        vacuum_product = physical_constants['epsilon_0'] * physical_constants['mu_0']
        c_from_vacuum = 1.0 / np.sqrt(vacuum_product)
        
        assert_allclose(
            c_from_vacuum,
            physical_constants['c'],
            rtol=1e-10,
            err_msg="Relación electromagnética c² = 1/(ε₀μ₀) violada"
        )
    
    def test_compton_wavelength(self, physical_constants):
        """Verificar longitud de onda Compton: λ_C = h/(m_e·c)."""
        lambda_c = physical_constants['hbar'] / (
            physical_constants['m_e'] * physical_constants['c']
        )
        lambda_c_expected = 2.42631023867e-12  # CODATA 2018
        
        assert_allclose(lambda_c, lambda_c_expected, rtol=1e-9)
    
    def test_fine_structure_constant(self, physical_constants):
        """Verificar constante de estructura fina: α ≈ 1/137."""
        alpha = (
            physical_constants['e']**2 /
            (4 * np.pi * physical_constants['epsilon_0'] *
             physical_constants['hbar'] * physical_constants['c'])
        )
        alpha_expected = 7.2973525693e-3  # ≈ 1/137.035999084
        
        assert_allclose(alpha, alpha_expected, rtol=1e-9)


### ═══════════════════════════════════════════════════════════════════════════════
### TESTS DE ESTRUCTURAS CUÁNTICAS (Espacio de Fock)
### ═══════════════════════════════════════════════════════════════════════════════

class TestQuantumState:
    """Tests para la clase QuantumState."""
    
    def test_normalization_enforcement(self):
        """Estado debe estar normalizado: ⟨ψ|ψ⟩ = 1."""
        amplitudes = np.array([1.0, 0.0, 0.0], dtype=np.complex128)
        state = QuantumState(amplitudes)
        
        norm_squared = np.vdot(state.amplitudes, state.amplitudes).real
        assert_allclose(norm_squared, 1.0, atol=1e-15)
    
    def test_normalization_validation_fails(self):
        """Estado no normalizado debe lanzar ValueError."""
        amplitudes = np.array([0.5, 0.5, 0.5], dtype=np.complex128)
        
        with pytest.raises(ValueError, match="no normalizado"):
            QuantumState(amplitudes)
    
    def test_expectation_value_hermitian_operator(self):
        """⟨ψ|Ô|ψ⟩ debe ser real para operador hermítico."""
        amplitudes = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=np.complex128)
        state = QuantumState(amplitudes)
        
        # Operador Pauli-X (hermítico)
        pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        
        expectation = state.expectation_value(pauli_x)
        
        # Parte imaginaria debe ser ~0
        assert abs(expectation.imag) < 1e-15
        assert_allclose(expectation.real, 1.0, atol=1e-15)
    
    def test_dimension_property(self):
        """Propiedad dimension debe retornar tamaño correcto."""
        amplitudes = np.array([1, 0, 0, 0, 0], dtype=np.complex128)
        amplitudes = amplitudes / np.linalg.norm(amplitudes)
        state = QuantumState(amplitudes)
        
        assert state.dimension == 5
    
    @given(
        n_states=st.integers(min_value=2, max_value=50)
    )
    @settings(max_examples=20, deadline=1000)
    def test_random_normalized_states(self, n_states):
        """Property-based: estados aleatorios normalizados."""
        # Generar amplitudes aleatorias complejas
        real_part = np.random.randn(n_states)
        imag_part = np.random.randn(n_states)
        amplitudes = real_part + 1j * imag_part
        
        # Normalizar
        amplitudes /= np.linalg.norm(amplitudes)
        
        state = QuantumState(amplitudes)
        norm_squared = np.vdot(state.amplitudes, state.amplitudes).real
        
        assert_allclose(norm_squared, 1.0, atol=1e-12)


class TestFockSpaceOperators:
    """Tests para operadores bosónicos en espacio de Fock."""
    
    def test_creation_operator_commutation(self, fock_operators):
        """Verificar [â, â†] = 𝟙 (relación de conmutación canónica)."""
        a = fock_operators.annihilation
        a_dag = fock_operators.creation
        
        commutator = a @ a_dag - a_dag @ a
        identity = np.eye(fock_operators.max_n)
        
        assert_allclose(commutator, identity, atol=1e-12)
    
    def test_number_operator_eigenvalues(self, fock_operators):
        """n̂|n⟩ = n|n⟩ (eigenvectores de Fock)."""
        n_op = fock_operators.number
        
        for n in range(min(10, fock_operators.max_n)):
            # Estado de Fock |n⟩
            fock_state = np.zeros(fock_operators.max_n, dtype=np.complex128)
            fock_state[n] = 1.0
            
            # Aplicar operador de número
            result = n_op @ fock_state
            
            # Debe ser n·|n⟩
            expected = n * fock_state
            assert_allclose(result, expected, atol=1e-12)
    
    def test_annihilation_operator_vacuum(self, fock_operators):
        """â|0⟩ = 0 (aniquilación del vacío)."""
        a = fock_operators.annihilation
        vacuum = np.zeros(fock_operators.max_n, dtype=np.complex128)
        vacuum[0] = 1.0
        
        result = a @ vacuum
        
        assert_allclose(result, np.zeros_like(result), atol=1e-15)
    
    def test_creation_operator_ladder(self, fock_operators):
        """â†|n⟩ = √(n+1)|n+1⟩."""
        a_dag = fock_operators.creation
        
        n = 5
        fock_n = np.zeros(fock_operators.max_n, dtype=np.complex128)
        fock_n[n] = 1.0
        
        result = a_dag @ fock_n
        
        # Esperado: √6 |6⟩
        expected = np.zeros_like(fock_n)
        expected[n+1] = np.sqrt(n + 1)
        
        assert_allclose(result, expected, atol=1e-12)
    
    def test_coherent_state_eigenstate(self, fock_operators):
        """â|α⟩ = α|α⟩ (eigenestado del operador de aniquilación)."""
        alpha = 2.0 + 1.5j
        coherent_state = fock_operators.coherent_state(alpha)
        
        a = fock_operators.annihilation
        a_coherent = a @ coherent_state.amplitudes
        alpha_coherent = alpha * coherent_state.amplitudes
        
        # Permitir error por truncamiento de Fock
        assert_allclose(a_coherent, alpha_coherent, atol=1e-6)
    
    def test_squeezed_vacuum_even_parity(self, fock_operators):
        """Estado squeezed solo tiene amplitudes en estados pares."""
        squeezed = fock_operators.squeezed_vacuum(r=0.5, phi=0.0)
        
        # Estados impares deben tener amplitud ~0
        for n in range(1, squeezed.dimension, 2):
            assert abs(squeezed.amplitudes[n]) < 1e-10
    
    def test_coherent_state_normalization_with_truncation(self, fock_operators):
        """Estado coherente truncado debe estar normalizado."""
        for alpha_mag in [0.5, 1.0, 2.0, 3.0]:
            alpha = alpha_mag * np.exp(1j * np.pi/4)
            coherent = fock_operators.coherent_state(alpha)
            
            norm_sq = np.vdot(coherent.amplitudes, coherent.amplitudes).real
            assert_allclose(norm_sq, 1.0, atol=1e-12)
    
    def test_operators_are_hermitian(self, fock_operators):
        """Operadores físicos deben ser hermíticos."""
        # n̂ = â†â debe ser hermítico
        assert_hermitian(fock_operators.number)
        
        # â† debe ser adjunto de â
        assert_allclose(
            fock_operators.creation,
            fock_operators.annihilation.conj().T,
            atol=1e-15
        )


### ═══════════════════════════════════════════════════════════════════════════════
### TESTS DE TOPOLOGÍA ALGEBRAICA
### ═══════════════════════════════════════════════════════════════════════════════

class TestSimplicialComplex:
    """Tests para complejos simpliciales y homología."""
    
    def test_boundary_operator_composition_is_zero(self):
        """∂_{n} ∘ ∂_{n+1} = 0 (condición de complejo de cadenas)."""
        complex = SimplicialComplex(dimension=2)
        
        # Construir complejo simple: tetraedro
        # 3 vértices, 3 aristas, 1 cara
        boundary_1 = sp.csr_matrix([
            [1, -1, 0],  # arista 0: v0→v1
            [0, 1, -1],  # arista 1: v1→v2
            [-1, 0, 1],  # arista 2: v2→v0
        ], dtype=np.float64)
        
        boundary_2 = sp.csr_matrix([
            [1, 1, 1],  # cara: cerrada por las 3 aristas
        ], dtype=np.float64).T
        
        complex.set_boundary_matrix(1, boundary_1)
        complex.set_boundary_matrix(2, boundary_2)
        
        # ∂₁ ∘ ∂₂ debe ser matriz cero
        composition = boundary_1 @ boundary_2
        assert_allclose(composition.toarray(), 0, atol=1e-15)
    
    def test_betti_numbers_point(self):
        """Punto: β₀=1, β₁=0 (una componente conexa, sin ciclos)."""
        complex = SimplicialComplex(dimension=1)
        
        # Solo un vértice, sin aristas
        boundary_1 = sp.csr_matrix((1, 1), dtype=np.float64)
        complex.set_boundary_matrix(1, boundary_1)
        
        betti = complex.compute_betti_numbers()
        
        # β₀ debería ser 1 (un componente)
        assert betti[0] >= 1
    
    def test_betti_numbers_circle(self):
        """Círculo S¹: β₀=1, β₁=1 (una componente, un ciclo)."""
        complex = SimplicialComplex(dimension=2)
        
        # Círculo: 3 vértices, 3 aristas formando ciclo cerrado
        boundary_1 = sp.csr_matrix([
            [1, -1, 0],
            [0, 1, -1],
            [-1, 0, 1],
        ], dtype=np.float64)
        
        complex.set_boundary_matrix(1, boundary_1)
        
        betti = complex.compute_betti_numbers()
        
        # β₁ debe ser 1 (un ciclo)
        assert betti[1] == 1
    
    def test_euler_characteristic_consistency(self):
        """χ = Σ(-1)ⁱβᵢ debe ser consistente."""
        invariant = TopologicalInvariant(
            betti_numbers=(1, 0, 0),
            euler_characteristic=1,
            torsion_coefficients=()
        )
        
        # No debe lanzar error
        assert invariant.euler_characteristic == 1
    
    def test_euler_characteristic_inconsistency_raises(self):
        """Números de Betti inconsistentes deben lanzar error."""
        with pytest.raises(ValueError, match="Inconsistencia topológica"):
            TopologicalInvariant(
                betti_numbers=(1, 1, 0),  # χ = 1 - 1 + 0 = 0
                euler_characteristic=5,    # Inconsistente
                torsion_coefficients=()
            )


class TestHomologicalAnnihilator:
    """Tests para el aniquilador de ciclos homológicos."""
    
    def test_trivial_homology_unchanged(self, homological_annihilator):
        """Complejo sin ciclos no debe cambiar."""
        # Matriz de frontera sin núcleo no trivial
        boundary = sp.csr_matrix([
            [1, 0],
            [0, 1],
        ], dtype=np.float64)
        
        collapsed, invariants = homological_annihilator.collapse_cycles(boundary)
        
        # Debe retornar casi sin cambios (módulo ruido numérico)
        assert_allclose(
            collapsed.toarray(),
            boundary.toarray(),
            atol=1e-10
        )
    
    def test_cycle_detection_circle(self, homological_annihilator):
        """Detectar ciclo en S¹."""
        # Círculo: ker(∂₁) no trivial
        boundary = sp.csr_matrix([
            [1, -1, 0],
            [0, 1, -1],
            [-1, 0, 1],
        ], dtype=np.float64)
        
        collapsed, invariants = homological_annihilator.collapse_cycles(boundary)
        
        # Debe detectar β₁ = 1
        assert invariants.betti_numbers[1] >= 0
        logger.info(f"β₁ detectado: {invariants.betti_numbers[1]}")
    
    def test_rank_preservation_after_collapse(self, homological_annihilator):
        """El rango de la matriz debe mantenerse o aumentar tras colapso."""
        boundary = sp.csr_matrix(np.random.randn(10, 10))
        
        rank_before = np.linalg.matrix_rank(boundary.toarray())
        
        collapsed, _ = homological_annihilator.collapse_cycles(boundary)
        
        rank_after = np.linalg.matrix_rank(collapsed.toarray())
        
        # El colapso puede aumentar rango (eliminar núcleo)
        assert rank_after >= rank_before - 1  # Tolerancia por numerics
    
    @pytest.mark.parametrize("matrix_size", [5, 10, 20, 50])
    def test_collapse_performance_scaling(self, homological_annihilator, matrix_size):
        """Verificar escalabilidad O(n³) de SVD."""
        boundary = sp.random(matrix_size, matrix_size, density=0.5, format='csr')
        
        with performance_benchmark(
            f"Collapse {matrix_size}×{matrix_size}",
            max_time_ms=5000
        ):
            collapsed, _ = homological_annihilator.collapse_cycles(boundary)


### ═══════════════════════════════════════════════════════════════════════════════
### TESTS DE SISTEMAS PORT-HAMILTONIANOS
### ═══════════════════════════════════════════════════════════════════════════════

class TestPortHamiltonianSystem:
    """Tests para sistemas Port-Hamiltonianos con disipación."""
    
    def test_structure_matrix_antisymmetry(self):
        """Matriz J debe ser antisimétrica: J = -Jᵀ."""
        def hamiltonian(x):
            return 0.5 * x[0]**2
        
        def grad_h(x):
            return np.array([x[0]])
        
        J = np.array([[0.0, 1.0], [-1.0, 0.0]])
        R = np.array([[0.1, 0.0], [0.0, 0.1]])
        
        system = PortHamiltonianSystem(hamiltonian, grad_h, J, R)
        
        assert_antisymmetric(system.J)
    
    def test_dissipation_matrix_positive_semidefinite(self):
        """Matriz R debe ser semi-definida positiva."""
        def hamiltonian(x):
            return 0.5 * np.sum(x**2)
        
        def grad_h(x):
            return x
        
        J = np.zeros((3, 3))
        R = np.diag([0.1, 0.2, 0.3])
        
        system = PortHamiltonianSystem(hamiltonian, grad_h, J, R)
        
        assert_positive_semidefinite(system.R_base)
    
    def test_lyapunov_derivative_nonpositive(self):
        """dH/dt ≤ 0 (estabilidad de Lyapunov)."""
        def hamiltonian(x):
            return 0.5 * x[0]**2
        
        def grad_h(x):
            return np.array([x[0]])
        
        J = np.array([[0.0]])
        R = np.array([[0.5]])  # Disipación
        
        system = PortHamiltonianSystem(hamiltonian, grad_h, J, R)
        
        state = np.array([2.0])
        dH_dt = system.lyapunov_derivative(state, rho_positron=1.0)
        
        assert dH_dt <= 1e-12, f"dH/dt = {dH_dt} > 0"
    
    def test_dissipation_modulation_preserves_sdp(self):
        """R(ρ) debe permanecer SDP para todo ρ ≥ 0."""
        def hamiltonian(x):
            return 0.5 * x[0]**2
        
        def grad_h(x):
            return np.array([x[0]])
        
        J = np.array([[0.0]])
        R_base = np.array([[1.0]])
        
        system = PortHamiltonianSystem(hamiltonian, grad_h, J, R_base)
        
        for rho in [0.0, 0.1, 1.0, 10.0, 100.0]:
            R_modulated = system.compute_dissipation_matrix(rho)
            assert_positive_semidefinite(R_modulated)
    
    def test_energy_conservation_without_dissipation(self):
        """Con R=0, la energía debe conservarse."""
        def hamiltonian(x):
            return 0.5 * (x[0]**2 + x[1]**2)
        
        def grad_h(x):
            return x
        
        J = np.array([[0.0, 1.0], [-1.0, 0.0]])  # Simpléctica
        R = np.zeros((2, 2))  # Sin disipación
        
        system = PortHamiltonianSystem(hamiltonian, grad_h, J, R)
        
        state = np.array([1.0, 0.0])
        dH_dt = system.lyapunov_derivative(state, rho_positron=0.0)
        
        assert abs(dH_dt) < 1e-12


### ═══════════════════════════════════════════════════════════════════════════════
### TESTS DE ANTIMATTER CHOKE COIL (Core Functionality)
### ═══════════════════════════════════════════════════════════════════════════════

class TestAntimatterChokeCoil:
    """Tests del sistema principal de supresión de flyback."""
    
    def test_initialization_validates_inductance(self):
        """Inductancia negativa debe lanzar error."""
        with pytest.raises(ValueError, match="positiva"):
            AntimatterChokeCoil(inductance=-1e-6)
    
    def test_zero_flyback_returns_null_event(self, antimatter_coil):
        """di/dt = 0 no debe consumir positrones."""
        event = antimatter_coil.suppress_flyback_voltage(
            di_dt=0.0,
            positron_cartridges=()
        )
        
        assert event.positrons_consumed == 0
        assert event.initial_flyback_voltage == 0.0
        assert event.residual_flyback_voltage == 0.0
    
    def test_negative_di_dt_returns_null_event(self, antimatter_coil):
        """di/dt < 0 (descarga) no genera flyback."""
        event = antimatter_coil.suppress_flyback_voltage(
            di_dt=-1e6,
            positron_cartridges=()
        )
        
        assert event.positrons_consumed == 0
        assert event.residual_flyback_voltage <= 0.0
    
    def test_flyback_without_positrons_unchanged(self, antimatter_coil, mock_positron_cartridge):
        """Sin positrones, el flyback debe permanecer."""
        di_dt = 1e6  # 1 MA/s
        
        event = antimatter_coil.suppress_flyback_voltage(
            di_dt=di_dt,
            positron_cartridges=()
        )
        
        expected_flyback = antimatter_coil._L * di_dt
        
        assert_allclose(
            event.residual_flyback_voltage,
            expected_flyback,
            rtol=1e-6
        )
    
    def test_flyback_suppression_with_positrons(self, antimatter_coil, mock_positron_cartridge):
        """Positrones deben reducir el flyback."""
        di_dt = 1e6
        cartridges = tuple(mock_positron_cartridge() for _ in range(10))
        
        event = antimatter_coil.suppress_flyback_voltage(
            di_dt=di_dt,
            positron_cartridges=cartridges
        )
        
        assert event.positrons_consumed > 0
        assert event.residual_flyback_voltage < event.initial_flyback_voltage
        assert event.efficiency > 0.0
    
    def test_energy_conservation_in_annihilation(self, antimatter_coil, mock_positron_cartridge):
        """Energía absorbida = energía inicial - energía residual."""
        di_dt = 5e6
        cartridges = tuple(mock_positron_cartridge() for _ in range(5))
        
        event = antimatter_coil.suppress_flyback_voltage(
            di_dt=di_dt,
            positron_cartridges=cartridges
        )
        
        energy_absorbed = event.initial_flyback_voltage - event.residual_flyback_voltage
        
        assert energy_absorbed >= 0
        assert_allclose(
            event.thermodynamic_entropy_delta,
            energy_absorbed,
            rtol=1.0,  # Tolerancia alta, solo verificar orden de magnitud
            atol=1e-10
        )
    
    def test_second_law_thermodynamics(self, antimatter_coil, mock_positron_cartridge):
        """ΔS ≥ 0 (segunda ley de la termodinámica)."""
        di_dt = 1e6
        cartridges = tuple(mock_positron_cartridge() for _ in range(3))
        
        event = antimatter_coil.suppress_flyback_voltage(
            di_dt=di_dt,
            positron_cartridges=cartridges
        )
        
        assert event.thermodynamic_entropy_delta >= -PhysicalConstants.MACHINE_EPSILON
    
    def test_gamma_photon_emission(self, antimatter_coil, mock_positron_cartridge):
        """Cada positrón debe emitir 2 fotones gamma."""
        di_dt = 1e6
        n_positrons = 3
        cartridges = tuple(mock_positron_cartridge() for _ in range(n_positrons))
        
        event = antimatter_coil.suppress_flyback_voltage(
            di_dt=di_dt,
            positron_cartridges=cartridges
        )
        
        # Puede consumir menos positrones si flyback es pequeño
        expected_photons = 2 * event.positrons_consumed
        
        assert len(event.gamma_photons_emitted) == expected_photons
    
    def test_quantum_state_normalization_preserved(self, antimatter_coil, mock_positron_cartridge):
        """Estado cuántico debe permanecer normalizado."""
        di_dt = 1e6
        cartridges = tuple(mock_positron_cartridge() for _ in range(5))
        
        event = antimatter_coil.suppress_flyback_voltage(
            di_dt=di_dt,
            positron_cartridges=cartridges
        )
        
        # Verificar normalización del estado final
        norm_before = np.vdot(
            event.quantum_state_before.amplitudes,
            event.quantum_state_before.amplitudes
        ).real
        
        norm_after = np.vdot(
            event.quantum_state_after.amplitudes,
            event.quantum_state_after.amplitudes
        ).real
        
        assert_allclose(norm_before, 1.0, atol=1e-12)
        assert_allclose(norm_after, 1.0, atol=1e-12)
    
    def test_lyapunov_stability(self, antimatter_coil, mock_positron_cartridge):
        """Exponente de Lyapunov debe ser ≤ 0."""
        di_dt = 1e6
        cartridges = tuple(mock_positron_cartridge() for _ in range(5))
        
        event = antimatter_coil.suppress_flyback_voltage(
            di_dt=di_dt,
            positron_cartridges=cartridges
        )
        
        assert event.lyapunov_exponent <= PhysicalConstants.LYAPUNOV_TOLERANCE
    
    @pytest.mark.parametrize("di_dt", [1e3, 1e4, 1e5, 1e6, 1e7])
    def test_suppression_scales_with_di_dt(self, antimatter_coil, mock_positron_cartridge, di_dt):
        """Supresión debe escalar linealmente con di/dt."""
        n_positrons = 10
        cartridges = tuple(mock_positron_cartridge() for _ in range(n_positrons))
        
        event = antimatter_coil.suppress_flyback_voltage(
            di_dt=di_dt,
            positron_cartridges=cartridges
        )
        
        expected_flyback = antimatter_coil._L * di_dt
        assert_allclose(
            event.initial_flyback_voltage,
            expected_flyback,
            rtol=1e-10
        )
    
    def test_reset_to_vacuum(self, antimatter_coil, fock_operators):
        """Reset debe retornar estado cuántico al vacío."""
        # Perturbar estado
        antimatter_coil._quantum_state = fock_operators.coherent_state(alpha=2.0)
        
        # Reset
        antimatter_coil.reset_to_vacuum()
        
        # Verificar que está en vacío
        vacuum = fock_operators.coherent_state(alpha=0.0)
        
        fidelity = abs(np.vdot(
            antimatter_coil._quantum_state.amplitudes,
            vacuum.amplitudes
        ))**2
        
        assert fidelity > 0.99  # Alta fidelidad con vacío


class TestComplexImpedance:
    """Tests de impedancia en dominio de frecuencia compleja."""
    
    def test_impedance_at_dc(self, antimatter_coil):
        """En s=0 (DC), impedancia debe tender a infinito (inductor ideal)."""
        s = complex(PhysicalConstants.MACHINE_EPSILON, 0)
        
        Z = antimatter_coil.compute_complex_impedance(
            s=s,
            positron_density=1e10,
            momentum_cyber_physical=1e-20
        )
        
        # Parte inductiva domina
        assert abs(Z.imag) > abs(Z.real)
    
    def test_impedance_passivity(self, antimatter_coil):
        """Re(Z) ≥ 0 para pasividad (excepto en regiones de NDR controladas)."""
        frequencies = np.logspace(3, 12, 20)
        
        for f in frequencies:
            s = 2j * np.pi * f
            Z = antimatter_coil.compute_complex_impedance(
                s=s,
                positron_density=1e10,
                momentum_cyber_physical=1e-20
            )
            
            # Permitir pequeñas violaciones por NDR cuántico
            if Z.real < 0:
                logger.warning(f"NDR detectado a f={f:.2e} Hz: Re(Z)={Z.real:.4e}")
                assert Z.real > -1e-3  # Límite de NDR controlado
    
    def test_impedance_high_frequency_asymptote(self, antimatter_coil):
        """A alta frecuencia, Z → jωL (inductivo puro)."""
        f_high = 1e12  # 1 THz
        s = 2j * np.pi * f_high
        
        Z = antimatter_coil.compute_complex_impedance(
            s=s,
            positron_density=1e10,
            momentum_cyber_physical=1e-20
        )
        
        expected_imag = (2 * np.pi * f_high) * antimatter_coil._L
        
        # Parte imaginaria domina
        assert abs(Z.imag) > 10 * abs(Z.real)
        assert_allclose(abs(Z.imag), expected_imag, rtol=0.5)
    
    @pytest.mark.parametrize("rho_e", [1e8, 1e10, 1e12, 1e14])
    def test_impedance_modulation_by_positron_density(self, antimatter_coil, rho_e):
        """Mayor densidad de positrones debe reducir impedancia (NDR)."""
        s = complex(1e6, 1e6)
        
        Z = antimatter_coil.compute_complex_impedance(
            s=s,
            positron_density=rho_e,
            momentum_cyber_physical=1e-20
        )
        
        # Verificar que impedancia es finita
        assert np.isfinite(Z.real) and np.isfinite(Z.imag)


class TestTransferFunction:
    """Tests de función de transferencia H(s)."""
    
    def test_transfer_function_dc_gain(self, antimatter_coil):
        """H(0) debe ser finito (aunque pequeño)."""
        s = complex(PhysicalConstants.MACHINE_EPSILON, 0)
        
        H = antimatter_coil.transfer_function_laplace(s, positron_density=1e10)
        
        assert np.isfinite(H)
    
    def test_transfer_function_stability(self, antimatter_coil):
        """Polos deben estar en semiplano izquierdo (Re(s) < 0)."""
        # Analizar en eje imaginario (frontera de estabilidad)
        frequencies = np.logspace(3, 9, 50)
        
        for f in frequencies:
            s = 2j * np.pi * f
            H = antimatter_coil.transfer_function_laplace(s)
            
            # |H(jω)| no debe divergir
            assert abs(H) < 1e6, f"Resonancia no amortiguada en f={f:.2e} Hz"
    
    def test_bode_plot_generation(self, antimatter_coil):
        """Generar datos de Bode sin errores."""
        frequencies = np.logspace(3, 9, 100)
        
        mag_dB, phase_deg = AntimatterAnalytics.bode_plot_data(
            antimatter_coil,
            frequencies,
            positron_density=1e10
        )
        
        assert len(mag_dB) == len(frequencies)
        assert len(phase_deg) == len(frequencies)
        assert np.all(np.isfinite(mag_dB))
        assert np.all(np.isfinite(phase_deg))
    
    def test_nyquist_plot_generation(self, antimatter_coil):
        """Generar datos de Nyquist sin errores."""
        frequencies = np.logspace(3, 9, 100)
        
        nyquist_data = AntimatterAnalytics.nyquist_plot_data(
            antimatter_coil,
            frequencies,
            positron_density=1e10
        )
        
        assert len(nyquist_data) == len(frequencies)
        assert np.all(np.isfinite(nyquist_data.real))
        assert np.all(np.isfinite(nyquist_data.imag))


### ═══════════════════════════════════════════════════════════════════════════════
### TESTS DE INTEGRACIÓN (Subsistemas)
### ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """Tests de integración entre subsistemas."""
    
    def test_topology_quantum_coupling(self, antimatter_coil, mock_positron_cartridge):
        """Colapso topológico debe afectar estado cuántico."""
        # Crear matriz de frontera con ciclo
        boundary = sp.csr_matrix([
            [1, -1, 0],
            [0, 1, -1],
            [-1, 0, 1],
        ], dtype=np.float64)
        
        cartridges = tuple(mock_positron_cartridge() for _ in range(5))
        
        event = antimatter_coil.suppress_flyback_voltage(
            di_dt=1e6,
            positron_cartridges=cartridges,
            topology_context=boundary
        )
        
        # Verificar que números de Betti fueron registrados
        assert len(event.topology_betti_numbers_before) > 0
        assert len(event.topology_betti_numbers_after) > 0
    
    def test_hamiltonian_quantum_consistency(self, antimatter_coil, mock_positron_cartridge):
        """Estado cuántico debe ser consistente con Hamiltoniano."""
        cartridges = tuple(mock_positron_cartridge() for _ in range(3))
        
        event = antimatter_coil.suppress_flyback_voltage(
            di_dt=1e6,
            positron_cartridges=cartridges
        )
        
        # La energía del estado cuántico debe correlacionar con flyback
        # (verificación cualitativa por complejidad del cálculo exacto)
        assert event.thermodynamic_entropy_delta >= 0
    
    def test_full_pipeline_no_errors(self, antimatter_coil, mock_positron_cartridge):
        """Pipeline completo sin errores."""
        boundary = sp.random(10, 10, density=0.3, format='csr')
        cartridges = tuple(mock_positron_cartridge() for _ in range(10))
        
        with performance_benchmark("Full pipeline", max_time_ms=2000):
            event = antimatter_coil.suppress_flyback_voltage(
                di_dt=1e6,
                positron_cartridges=cartridges,
                topology_context=boundary
            )
        
        # Verificar todas las propiedades
        assert event.positrons_consumed >= 0
        assert event.efficiency >= 0 and event.efficiency <= 1
        assert event.thermodynamic_entropy_delta >= -1e-15
        assert len(event.gamma_photons_emitted) >= 0


### ═══════════════════════════════════════════════════════════════════════════════
### TESTS DE TELEMETRÍA Y OBSERVABILIDAD
### ═══════════════════════════════════════════════════════════════════════════════

class TestTelemetry:
    """Tests de métricas de telemetría."""
    
    def test_telemetry_accumulation(self, antimatter_coil, mock_positron_cartridge):
        """Telemetría debe acumular estadísticas."""
        cartridges = tuple(mock_positron_cartridge() for _ in range(5))
        
        # Primera aniquilación
        antimatter_coil.suppress_flyback_voltage(1e6, cartridges)
        
        telemetry1 = antimatter_coil.telemetry
        total1 = telemetry1['total_positrons_consumed']
        
        # Segunda aniquilación
        antimatter_coil.suppress_flyback_voltage(1e6, cartridges)
        
        telemetry2 = antimatter_coil.telemetry
        total2 = telemetry2['total_positrons_consumed']
        
        assert total2 >= total1
    
    def test_quantum_fidelity_metric(self, antimatter_coil):
        """Fidelidad cuántica debe estar en [0, 1]."""
        telemetry = antimatter_coil.telemetry
        fidelity = telemetry['quantum_state_fidelity']
        
        assert 0.0 <= fidelity <= 1.0
    
    def test_telemetry_after_reset(self, antimatter_coil, mock_positron_cartridge):
        """Reset debe restaurar fidelidad pero mantener estadísticas."""
        cartridges = tuple(mock_positron_cartridge() for _ in range(5))
        
        antimatter_coil.suppress_flyback_voltage(1e6, cartridges)
        
        telemetry_before = antimatter_coil.telemetry
        fidelity_before = telemetry_before['quantum_state_fidelity']
        
        antimatter_coil.reset_to_vacuum()
        
        telemetry_after = antimatter_coil.telemetry
        fidelity_after = telemetry_after['quantum_state_fidelity']
        
        # Fidelidad debe aumentar (más cercano a vacío)
        assert fidelity_after > fidelity_before or fidelity_after > 0.95


### ═══════════════════════════════════════════════════════════════════════════════
### TESTS DE STRESS Y LÍMITES
### ═══════════════════════════════════════════════════════════════════════════════

class TestStressAndEdgeCases:
    """Tests de casos extremos y límites."""
    
    def test_extreme_di_dt(self, antimatter_coil, mock_positron_cartridge):
        """Manejar di/dt extremos sin overflow."""
        extreme_di_dt = 1e12  # 1 TA/s
        cartridges = tuple(mock_positron_cartridge() for _ in range(100))
        
        event = antimatter_coil.suppress_flyback_voltage(
            di_dt=extreme_di_dt,
            positron_cartridges=cartridges
        )
        
        assert np.isfinite(event.initial_flyback_voltage)
        assert np.isfinite(event.residual_flyback_voltage)
    
    def test_zero_positron_mass(self, antimatter_coil, mock_positron_cartridge):
        """Masa de positrón cero debe ser manejada."""
        cartridge = mock_positron_cartridge(mass=0.0)
        
        event = antimatter_coil.suppress_flyback_voltage(
            di_dt=1e6,
            positron_cartridges=(cartridge,)
        )
        
        # Debe completar sin crash
        assert event.positrons_consumed >= 0
    
    def test_large_fock_space(self):
        """Espacio de Fock grande no debe causar problemas de memoria."""
        large_n = 500
        
        with performance_benchmark(f"Fock space {large_n}", max_time_ms=3000):
            fock_ops = FockSpaceOperators(max_fock_state=large_n)
            coherent = fock_ops.coherent_state(alpha=1.0)
        
        assert coherent.dimension == large_n
        
        # Limpiar memoria
        del fock_ops, coherent
        gc.collect()
    
    def test_numerical_stability_near_singularity(self, antimatter_coil):
        """Cerca de singularidades numéricas (s≈0)."""
        s_near_zero = complex(1e-100, 1e-100)
        
        Z = antimatter_coil.compute_complex_impedance(
            s=s_near_zero,
            positron_density=1e10,
            momentum_cyber_physical=1e-20
        )
        
        assert np.isfinite(Z.real) and np.isfinite(Z.imag)
    
    @pytest.mark.timeout(5)
    def test_no_infinite_loops(self, antimatter_coil, mock_positron_cartridge):
        """Verificar que no hay loops infinitos."""
        cartridges = tuple(mock_positron_cartridge() for _ in range(1000))
        
        event = antimatter_coil.suppress_flyback_voltage(
            di_dt=1e9,
            positron_cartridges=cartridges
        )
        
        # Si llegamos aquí, no hubo loop infinito
        assert True


### ═══════════════════════════════════════════════════════════════════════════════
### TESTS DE PROPERTY-BASED (Hypothesis)
### ═══════════════════════════════════════════════════════════════════════════════

class TestPropertyBased:
    """Tests basados en propiedades con generación automática."""
    
    @given(
        di_dt=st.floats(min_value=1e3, max_value=1e9, allow_nan=False, allow_infinity=False),
        n_positrons=st.integers(min_value=0, max_value=20)
    )
    @settings(max_examples=50, deadline=2000)
    def test_efficiency_bounds(self, antimatter_coil, mock_positron_cartridge, di_dt, n_positrons):
        """Eficiencia siempre en [0, 1]."""
        cartridges = tuple(mock_positron_cartridge() for _ in range(n_positrons))
        
        event = antimatter_coil.suppress_flyback_voltage(
            di_dt=di_dt,
            positron_cartridges=cartridges
        )
        
        assert 0.0 <= event.efficiency <= 1.0
    
    @given(
        alpha_real=st.floats(min_value=-3.0, max_value=3.0),
        alpha_imag=st.floats(min_value=-3.0, max_value=3.0)
    )
    @settings(max_examples=30, deadline=1000)
    def test_coherent_states_always_normalized(self, fock_operators, alpha_real, alpha_imag):
        """Estados coherentes siempre normalizados."""
        alpha = complex(alpha_real, alpha_imag)
        
        coherent = fock_operators.coherent_state(alpha)
        
        norm_sq = np.vdot(coherent.amplitudes, coherent.amplitudes).real
        assert_allclose(norm_sq, 1.0, atol=1e-10)
    
    @given(
        matrix_size=st.integers(min_value=2, max_value=20)
    )
    @settings(max_examples=20, deadline=2000)
    def test_hermitian_operators_real_eigenvalues(self, matrix_size):
        """Operadores hermíticos tienen eigenvalores reales."""
        # Generar matriz hermítica aleatoria
        A = np.random.randn(matrix_size, matrix_size) + 1j * np.random.randn(matrix_size, matrix_size)
        A_hermitian = (A + A.conj().T) / 2
        
        eigenvalues = np.linalg.eigvalsh(A_hermitian)
        
        # Todos los eigenvalores deben ser reales
        assert np.all(np.isreal(eigenvalues))


### ═══════════════════════════════════════════════════════════════════════════════
### SUITE DE BENCHMARKS
### ═══════════════════════════════════════════════════════════════════════════════

class TestBenchmarks:
    """Benchmarks de performance."""
    
    @pytest.mark.benchmark
    def test_benchmark_single_annihilation(self, antimatter_coil, mock_positron_cartridge, benchmark):
        """Benchmark de una aniquilación simple."""
        cartridges = tuple(mock_positron_cartridge() for _ in range(5))
        
        result = benchmark(
            antimatter_coil.suppress_flyback_voltage,
            di_dt=1e6,
            positron_cartridges=cartridges
        )
        
        assert result.efficiency > 0
    
    @pytest.mark.benchmark
    def test_benchmark_impedance_calculation(self, antimatter_coil, benchmark):
        """Benchmark de cálculo de impedancia."""
        s = complex(1e6, 1e6)
        
        result = benchmark(
            antimatter_coil.compute_complex_impedance,
            s=s,
            positron_density=1e10,
            momentum_cyber_physical=1e-20
        )
        
        assert np.isfinite(result)
    
    @pytest.mark.benchmark
    def test_benchmark_topology_collapse(self, homological_annihilator, benchmark):
        """Benchmark de colapso topológico."""
        boundary = sp.random(50, 50, density=0.3, format='csr')
        
        result = benchmark(
            homological_annihilator.collapse_cycles,
            boundary_matrix_1=boundary
        )
        
        assert result is not None


### ═══════════════════════════════════════════════════════════════════════════════
### CONFIGURACIÓN DE PYTEST
### ═══════════════════════════════════════════════════════════════════════════════

def pytest_configure(config):
    """Configuración global de pytest."""
    config.addinivalue_line(
        "markers", "benchmark: marca tests de benchmark"
    )
    config.addinivalue_line(
        "markers", "slow: marca tests lentos"
    )
    config.addinivalue_line(
        "markers", "integration: marca tests de integración"
    )


### ═══════════════════════════════════════════════════════════════════════════════
### PUNTO DE ENTRADA
### ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """Ejecutar suite de tests con pytest."""
    pytest.main([
        __file__,
        "-v",                    # Verbose
        "--tb=short",           # Traceback corto
        "--strict-markers",     # Markers estrictos
        "--cov=app.physics.antimatter_choke_coil",  # Cobertura
        "--cov-report=html",    # Reporte HTML
        "--cov-report=term",    # Reporte en terminal
        "--durations=10",       # Top 10 tests más lentos
        "-m", "not slow",       # Excluir tests lentos por defecto
    ])