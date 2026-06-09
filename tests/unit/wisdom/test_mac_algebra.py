# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo de Pruebas: MAC Algebra Test Suite                                    ║
║ Ubicación: tests/wisdom/test_mac_algebra.py                                  ║
║ Versión: 3.0.0-Dagger-Compact-Category-Test-Suite                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Suite de Pruebas Rigurosas para Validación de:
───────────────────────────────────────────────
1. Morfismos CPTP (Completitud Positiva, Preservación de Traza)
2. Categorías Compactas con Daga (Composición, Adjuntos)
3. Retículo Ortomodular (Lógica Cuántica, Proyectores)
4. Teoría Modular de Tomita-Takesaki (Conjugación, Grupo Modular)
5. Álgebras de von Neumann (Factores, Dualidad)

Metodología de Testing:
───────────────────────
- Pruebas unitarias (componentes aislados)
- Pruebas de axiomas matemáticos
- Pruebas de propiedades algebraicas
- Pruebas de convergencia
- Pruebas de teoría de categorías
- Pruebas de lógica cuántica

Referencias:
────────────
- Murray & von Neumann (1936): "On rings of operators"
- Tomita (1967): "Standard forms of von Neumann algebras"
- Birkhoff & von Neumann (1936): "The logic of quantum mechanics"
- Abramsky & Coecke (2004): "A categorical semantics of quantum protocols"
═══════════════════════════════════════════════════════════════════════════════
"""

import pytest
import numpy as np
import scipy.linalg as la
from typing import List, Tuple, Dict
from hypothesis import given, strategies as st, settings, assume
import logging

# Módulos bajo prueba
import sys
sys.path.insert(0, '../../..')

from app.wisdom.mac_algebra import (
    CPTPMorphism,
    ComposedMorphism,
    OrthomodularLattice,
    TomitaTakesakiTheory,
    NonCommutativeAlgebraError,
    TraceAnomalyError,
    OrthomodularConvergenceError,
    ModularConjugationError,
    VonNeumannFactorType,
    ModularTheoryData
)

from app.wisdom.atomic_knowledge_matrix import (
    AtomicDensityMatrix,
    create_quantum_mac_state,
    NumericalInstabilityError
)

logger = logging.getLogger("MAC.Algebra.Tests")


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES GLOBALES Y UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def pauli_operators() -> Dict[str, np.ndarray]:
    """Operadores de Pauli estándar."""
    return {
        'I': np.array([[1, 0], [0, 1]], dtype=np.complex128),
        'X': np.array([[0, 1], [1, 0]], dtype=np.complex128),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        'Z': np.array([[1, 0], [0, -1]], dtype=np.complex128)
    }


@pytest.fixture
def hadamard_gate() -> np.ndarray:
    """Puerta de Hadamard."""
    return np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)


def create_unitary_channel(U: np.ndarray) -> CPTPMorphism:
    """Crea canal unitario CPTP."""
    return CPTPMorphism(kraus_operators=tuple([U]))


def create_depolarizing_channel(dimension: int, p: float) -> CPTPMorphism:
    """Crea canal despolarizante CPTP."""
    if dimension != 2:
        raise NotImplementedError("Solo implementado para d=2")
    
    pauli_ops = [
        np.array([[1, 0], [0, 1]], dtype=np.complex128),
        np.array([[0, 1], [1, 0]], dtype=np.complex128),
        np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        np.array([[1, 0], [0, -1]], dtype=np.complex128)
    ]
    
    kraus_ops = tuple([
        np.sqrt(1 - 3*p/4) * pauli_ops[0],
        np.sqrt(p/4) * pauli_ops[1],
        np.sqrt(p/4) * pauli_ops[2],
        np.sqrt(p/4) * pauli_ops[3]
    ])
    
    return CPTPMorphism(kraus_operators=kraus_ops)


def create_projector(dimension: int, subspace_indices: List[int]) -> np.ndarray:
    """Crea proyector ortogonal sobre subespacio."""
    P = np.zeros((dimension, dimension), dtype=np.complex128)
    for i in subspace_indices:
        P[i, i] = 1.0
    return P


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1: PRUEBAS DE CPTP MORPHISM
# ══════════════════════════════════════════════════════════════════════════════

class TestCPTPMorphism:
    """Suite de pruebas para morfismos CPTP."""
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Validación de Axiomas
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_cptp_validates_identity_resolution(self, pauli_operators):
        """Debe validar resolución de identidad Σₖ Mₖ†Mₖ = I."""
        # Canal válido (unitario)
        U = pauli_operators['X']
        channel = create_unitary_channel(U)
        
        # Debe crearse sin errores
        assert channel.dimension == 2
        assert channel.kraus_rank == 1
    
    def test_cptp_rejects_invalid_identity_resolution(self):
        """Debe rechazar operadores que no resuelven identidad."""
        # Operadores inválidos
        M1 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        
        with pytest.raises(TraceAnomalyError, match="identidad"):
            CPTPMorphism(kraus_operators=tuple([M1]))
    
    def test_cptp_rejects_empty_operators(self):
        """Debe rechazar conjunto vacío de operadores."""
        with pytest.raises(NonCommutativeAlgebraError, match="vacío"):
            CPTPMorphism(kraus_operators=tuple())
    
    def test_cptp_validates_consistent_dimensions(self):
        """Debe validar dimensiones consistentes."""
        # Operadores con dimensiones inconsistentes
        M1 = np.eye(2, dtype=np.complex128)
        M2 = np.eye(3, dtype=np.complex128)
        
        with pytest.raises(ValueError, match="inconsistente"):
            CPTPMorphism(kraus_operators=tuple([M1, M2]))
    
    def test_cptp_rejects_non_square_operators(self):
        """Debe rechazar operadores no cuadrados."""
        M1 = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.complex128)
        
        with pytest.raises(ValueError, match="cuadrado"):
            CPTPMorphism(kraus_operators=tuple([M1]))
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Aplicación Forward
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_cptp_apply_preserves_trace(self, pauli_operators):
        """Aplicación debe preservar traza."""
        U = pauli_operators['Z']
        channel = create_unitary_channel(U)
        
        rho = create_quantum_mac_state(dimension=2, purity=0.7, seed=42)
        rho_out = channel.apply(rho.matrix)
        
        trace_in = np.trace(rho.matrix).real
        trace_out = np.trace(rho_out).real
        
        assert np.isclose(trace_out, trace_in, atol=1e-10), \
            f"Traza no preservada: {trace_in} → {trace_out}"
    
    def test_cptp_apply_preserves_hermiticity(self, hadamard_gate):
        """Aplicación debe preservar hermiticidad."""
        channel = create_unitary_channel(hadamard_gate)
        
        rho = create_quantum_mac_state(dimension=2, purity=0.8, seed=42)
        rho_out = channel.apply(rho.matrix)
        
        hermiticity_error = la.norm(rho_out - rho_out.conj().T, ord='fro')
        
        assert hermiticity_error < 1e-10, \
            f"Hermiticidad no preservada: {hermiticity_error}"
    
    def test_cptp_apply_preserves_positivity(self, pauli_operators):
        """Aplicación debe preservar positividad."""
        U = pauli_operators['Y']
        channel = create_unitary_channel(U)
        
        rho = create_quantum_mac_state(dimension=2, purity=0.6, seed=42)
        rho_out = channel.apply(rho.matrix)
        
        eigenvalues = la.eigvalsh(rho_out)
        
        assert np.all(eigenvalues >= -1e-10), \
            f"Positividad no preservada: {eigenvalues}"
    
    def test_cptp_apply_unitary_channel(self, pauli_operators):
        """Canal unitario debe implementar U ρ U†."""
        U = pauli_operators['X']
        channel = create_unitary_channel(U)
        
        rho = create_quantum_mac_state(dimension=2, purity=0.9, seed=42)
        
        # Aplicar canal
        rho_out_channel = channel.apply(rho.matrix)
        
        # Aplicar directamente U ρ U†
        rho_out_direct = U @ rho.matrix @ U.conj().T
        
        difference = la.norm(rho_out_channel - rho_out_direct, ord='fro')
        
        assert difference < 1e-12, \
            f"Canal unitario inconsistente: {difference}"
    
    def test_cptp_apply_depolarizing_channel(self):
        """Canal despolarizante debe mezclar hacia I/d."""
        p = 0.5
        channel = create_depolarizing_channel(dimension=2, p=p)
        
        # Estado puro inicial
        psi = np.array([1, 0], dtype=np.complex128)
        rho = np.outer(psi, psi.conj())
        
        rho_out = channel.apply(rho)
        
        # Verificar que se mezcla
        purity_in = np.trace(rho @ rho).real
        purity_out = np.trace(rho_out @ rho_out).real
        
        assert purity_out < purity_in, \
            "Canal despolarizante no reduce pureza"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Aplicación Adjunta
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_cptp_adjoint_apply_heisenberg_picture(self, pauli_operators):
        """Adjunto debe implementar evolución de Heisenberg."""
        U = pauli_operators['Z']
        channel = create_unitary_channel(U)
        
        # Observable
        observable = pauli_operators['X']
        
        # Aplicar adjunto
        obs_out = channel.adjoint_apply(observable)
        
        # Evolución de Heisenberg: U† A U
        obs_expected = U.conj().T @ observable @ U
        
        difference = la.norm(obs_out - obs_expected, ord='fro')
        
        assert difference < 1e-12, \
            f"Adjunto inconsistente: {difference}"
    
    def test_cptp_adjoint_preserves_hermiticity(self, pauli_operators):
        """Adjunto debe preservar hermiticidad de observables."""
        U = pauli_operators['Y']
        channel = create_unitary_channel(U)
        
        observable = pauli_operators['Z']
        obs_out = channel.adjoint_apply(observable)
        
        hermiticity_error = la.norm(obs_out - obs_out.conj().T, ord='fro')
        
        assert hermiticity_error < 1e-10, \
            f"Hermiticidad del observable no preservada: {hermiticity_error}"
    
    def test_cptp_adjoint_schrödinger_heisenberg_duality(self, hadamard_gate):
        """Dualidad: ⟨ℰ(ρ)|A⟩ = ⟨ρ|ℰ†(A)⟩."""
        channel = create_unitary_channel(hadamard_gate)
        
        rho = create_quantum_mac_state(dimension=2, purity=0.7, seed=42)
        observable = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        
        # Schrödinger: Tr(ℰ(ρ) A)
        rho_evolved = channel.apply(rho.matrix)
        expectation_schrödinger = np.trace(rho_evolved @ observable).real
        
        # Heisenberg: Tr(ρ ℰ†(A))
        obs_evolved = channel.adjoint_apply(observable)
        expectation_heisenberg = np.trace(rho.matrix @ obs_evolved).real
        
        assert np.isclose(expectation_schrödinger, expectation_heisenberg, atol=1e-10), \
            f"Dualidad violada: {expectation_schrödinger} vs {expectation_heisenberg}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Propiedades del Canal
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_cptp_is_unitary_detects_unitary_channel(self, pauli_operators):
        """Debe detectar canales unitarios correctamente."""
        U = pauli_operators['X']
        channel = create_unitary_channel(U)
        
        assert channel.is_unitary(), "Canal unitario no detectado"
    
    def test_cptp_is_unitary_rejects_non_unitary(self):
        """Debe rechazar canales no unitarios."""
        channel = create_depolarizing_channel(dimension=2, p=0.3)
        
        assert not channel.is_unitary(), "Canal no unitario detectado como unitario"
    
    def test_cptp_is_unital_pauli_channel(self, pauli_operators):
        """Canal de Pauli debe ser unital."""
        kraus_ops = tuple([
            np.sqrt(0.25) * pauli_operators['I'],
            np.sqrt(0.25) * pauli_operators['X'],
            np.sqrt(0.25) * pauli_operators['Y'],
            np.sqrt(0.25) * pauli_operators['Z']
        ])
        
        channel = CPTPMorphism(kraus_operators=kraus_ops)
        
        assert channel.is_unital(), "Canal de Pauli no es unital"
    
    def test_cptp_compute_choi_matrix_properties(self, pauli_operators):
        """Matriz de Choi debe ser hermitiana y positiva."""
        U = pauli_operators['Z']
        channel = create_unitary_channel(U)
        
        choi = channel.compute_choi_matrix()
        
        # Hermiticidad
        hermiticity_error = la.norm(choi - choi.conj().T, ord='fro')
        assert hermiticity_error < 1e-10, "Choi no hermitiana"
        
        # Positividad
        eigenvalues = la.eigvalsh(choi)
        assert np.all(eigenvalues >= -1e-10), "Choi no positiva"
    
    def test_cptp_dimension_property(self, pauli_operators):
        """Propiedad dimension debe retornar dimensión correcta."""
        U = pauli_operators['X']
        channel = create_unitary_channel(U)
        
        assert channel.dimension == 2, "Dimensión incorrecta"
    
    def test_cptp_kraus_rank_property(self):
        """Propiedad kraus_rank debe retornar número de operadores."""
        channel = create_depolarizing_channel(dimension=2, p=0.2)
        
        assert channel.kraus_rank == 4, "Rango de Kraus incorrecto"


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2: PRUEBAS DE COMPOSICIÓN DE MORFISMOS
# ══════════════════════════════════════════════════════════════════════════════

class TestComposedMorphism:
    """Suite de pruebas para composición de morfismos."""
    
    def test_composition_associativity(self, pauli_operators):
        """Composición debe ser asociativa: (f∘g)∘h = f∘(g∘h)."""
        U1 = pauli_operators['X']
        U2 = pauli_operators['Y']
        U3 = pauli_operators['Z']
        
        channel1 = create_unitary_channel(U1)
        channel2 = create_unitary_channel(U2)
        channel3 = create_unitary_channel(U3)
        
        # (channel1 ∘ channel2) ∘ channel3
        comp_left = channel1.compose(channel2).compose(channel3)
        
        # channel1 ∘ (channel2 ∘ channel3)
        comp_right = channel1.compose(channel2.compose(channel3))
        
        rho = create_quantum_mac_state(dimension=2, purity=0.8, seed=42)
        
        rho_left = comp_left.apply(rho.matrix)
        rho_right = comp_right.apply(rho.matrix)
        
        difference = la.norm(rho_left - rho_right, ord='fro')
        
        assert difference < 1e-10, \
            f"Asociatividad violada: {difference}"
    
    def test_composition_preserves_cptp(self, pauli_operators):
        """Composición de canales CPTP debe ser CPTP."""
        U1 = pauli_operators['X']
        U2 = pauli_operators['Z']
        
        channel1 = create_unitary_channel(U1)
        channel2 = create_unitary_channel(U2)
        
        composed = channel1.compose(channel2)
        
        rho = create_quantum_mac_state(dimension=2, purity=0.7, seed=42)
        rho_out = composed.apply(rho.matrix)
        
        # Verificar traza
        trace = np.trace(rho_out).real
        assert np.isclose(trace, 1.0, atol=1e-10), \
            f"Composición no preserva traza: {trace}"
        
        # Verificar positividad
        eigenvalues = la.eigvalsh(rho_out)
        assert np.all(eigenvalues >= -1e-10), \
            "Composición no preserva positividad"
    
    def test_composition_adjoint_reverses_order(self, pauli_operators):
        """Adjunto de composición: (f∘g)† = g†∘f†."""
        U1 = pauli_operators['X']
        U2 = pauli_operators['Y']
        
        channel1 = create_unitary_channel(U1)
        channel2 = create_unitary_channel(U2)
        
        composed = channel1.compose(channel2)
        
        observable = pauli_operators['Z']
        
        # (f∘g)†(A)
        obs_composed = composed.adjoint_apply(observable)
        
        # g†(f†(A))
        obs_reversed = channel2.adjoint_apply(channel1.adjoint_apply(observable))
        
        difference = la.norm(obs_composed - obs_reversed, ord='fro')
        
        assert difference < 1e-12, \
            f"Adjunto de composición incorrecto: {difference}"


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3: PRUEBAS DE RETÍCULO ORTOMODULAR
# ══════════════════════════════════════════════════════════════════════════════

class TestOrthomodularLattice:
    """Suite de pruebas para retículo ortomodular."""
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Validación de Proyectores
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_validate_projector_accepts_valid(self):
        """Debe aceptar proyectores válidos."""
        P = create_projector(dimension=3, subspace_indices=[0, 1])
        
        # No debe lanzar error
        OrthomodularLattice.validate_projector(P)
    
    def test_validate_projector_rejects_non_idempotent(self):
        """Debe rechazar matrices no idempotentes."""
        # Matriz que no satisface P² = P
        M = np.array([[1, 1], [0, 0]], dtype=np.complex128)
        
        with pytest.raises(ValueError, match="idempotente"):
            OrthomodularLattice.validate_projector(M)
    
    def test_validate_projector_rejects_non_hermitian(self):
        """Debe rechazar matrices no hermitianas."""
        # Matriz idempotente pero no hermitiana
        M = np.array([[1, 1], [0, 0]], dtype=np.complex128)
        # M² = M pero M ≠ M†
        
        with pytest.raises(ValueError, match="hermitian"):
            OrthomodularLattice.validate_projector(M)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Conjunción Ortomodular
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_quantum_conjunction_commuting_projectors(self):
        """Conjunción de proyectores que conmutan debe ser el producto."""
        P_A = create_projector(dimension=3, subspace_indices=[0, 1])
        P_B = create_projector(dimension=3, subspace_indices=[0])
        
        P_conjunction = OrthomodularLattice.quantum_conjunction(P_A, P_B)
        
        # Para proyectores que conmutan: P_A ∧ P_B = P_A P_B
        P_expected = P_A @ P_B
        
        difference = la.norm(P_conjunction - P_expected, ord='fro')
        
        assert difference < 1e-10, \
            f"Conjunción incorrecta para proyectores conmutantes: {difference}"
    
    def test_quantum_conjunction_is_projector(self):
        """Conjunción debe resultar en proyector."""
        P_A = create_projector(dimension=3, subspace_indices=[0, 1])
        P_B = create_projector(dimension=3, subspace_indices=[1, 2])
        
        P_conjunction = OrthomodularLattice.quantum_conjunction(P_A, P_B)
        
        # Verificar que es proyector
        OrthomodularLattice.validate_projector(P_conjunction, tolerance=1e-8)
    
    def test_quantum_conjunction_convergence(self):
        """Conjunción debe converger en número finito de iteraciones."""
        P_A = create_projector(dimension=4, subspace_indices=[0, 1])
        P_B = create_projector(dimension=4, subspace_indices=[1, 2])
        
        # Debe converger sin lanzar error
        P_conjunction = OrthomodularLattice.quantum_conjunction(
            P_A, P_B, max_iter=1000, tolerance=1e-10
        )
        
        assert P_conjunction is not None
    
    def test_quantum_conjunction_divergence_raises_error(self):
        """Debe lanzar error si no converge."""
        # Diseñar proyectores que no convergen rápidamente
        P_A = create_projector(dimension=2, subspace_indices=[0])
        P_B = create_projector(dimension=2, subspace_indices=[1])
        
        # Con max_iter muy bajo
        with pytest.raises(OrthomodularConvergenceError):
            OrthomodularLattice.quantum_conjunction(P_A, P_B, max_iter=1)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Complemento Ortomodular
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_quantum_complement_involution(self):
        """Complemento debe ser involutivo: (P⊥)⊥ = P."""
        P = create_projector(dimension=3, subspace_indices=[0, 1])
        
        P_complement = OrthomodularLattice.quantum_complement(P)
        P_double_complement = OrthomodularLattice.quantum_complement(P_complement)
        
        difference = la.norm(P_double_complement - P, ord='fro')
        
        assert difference < 1e-12, \
            f"Complemento no involutivo: {difference}"
    
    def test_quantum_complement_orthogonality(self):
        """P y P⊥ deben ser ortogonales: P P⊥ = 0."""
        P = create_projector(dimension=3, subspace_indices=[0, 1])
        P_complement = OrthomodularLattice.quantum_complement(P)
        
        product = P @ P_complement
        
        norm_product = la.norm(product, ord='fro')
        
        assert norm_product < 1e-12, \
            f"P y P⊥ no ortogonales: {norm_product}"
    
    def test_quantum_complement_completeness(self):
        """P + P⊥ = I."""
        P = create_projector(dimension=3, subspace_indices=[0, 1])
        P_complement = OrthomodularLattice.quantum_complement(P)
        
        sum_projectors = P + P_complement
        identity = np.eye(3, dtype=np.complex128)
        
        difference = la.norm(sum_projectors - identity, ord='fro')
        
        assert difference < 1e-12, \
            f"P + P⊥ ≠ I: {difference}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Disyunción Ortomodular
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_quantum_disjunction_is_projector(self):
        """Disyunción debe resultar en proyector."""
        P_A = create_projector(dimension=3, subspace_indices=[0])
        P_B = create_projector(dimension=3, subspace_indices=[1])
        
        P_disjunction = OrthomodularLattice.quantum_disjunction(P_A, P_B)
        
        # Verificar que es proyector
        OrthomodularLattice.validate_projector(P_disjunction, tolerance=1e-8)
    
    def test_quantum_disjunction_commutativity(self):
        """Disyunción debe ser conmutativa: P_A ∨ P_B = P_B ∨ P_A."""
        P_A = create_projector(dimension=3, subspace_indices=[0])
        P_B = create_projector(dimension=3, subspace_indices=[1])
        
        P_AB = OrthomodularLattice.quantum_disjunction(P_A, P_B)
        P_BA = OrthomodularLattice.quantum_disjunction(P_B, P_A)
        
        difference = la.norm(P_AB - P_BA, ord='fro')
        
        assert difference < 1e-10, \
            f"Disyunción no conmutativa: {difference}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Compatibilidad
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_are_compatible_orthogonal_projectors(self):
        """Proyectores ortogonales deben conmutar."""
        P_A = create_projector(dimension=3, subspace_indices=[0])
        P_B = create_projector(dimension=3, subspace_indices=[1])
        
        assert OrthomodularLattice.are_compatible(P_A, P_B), \
            "Proyectores ortogonales no conmutan"
    
    def test_are_compatible_general_projectors(self):
        """Proyectores generales pueden no conmutar."""
        # Crear proyectores no ortogonales
        dim = 2
        theta = np.pi / 6
        
        # P_A = |0⟩⟨0|
        P_A = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        
        # P_B = |+θ⟩⟨+θ| donde |+θ⟩ = cos(θ)|0⟩ + sin(θ)|1⟩
        psi_theta = np.array([np.cos(theta), np.sin(theta)], dtype=np.complex128)
        P_B = np.outer(psi_theta, psi_theta.conj())
        
        # Verificar conmutatividad
        commutator = OrthomodularLattice.commutator(P_A, P_B)
        
        # No deberían conmutar perfectamente
        if la.norm(commutator, ord='fro') > 1e-10:
            assert not OrthomodularLattice.are_compatible(P_A, P_B)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 4: PRUEBAS DE TEORÍA MODULAR DE TOMITA-TAKESAKI
# ══════════════════════════════════════════════════════════════════════════════

class TestTomitaTakesakiTheory:
    """Suite de pruebas para teoría modular."""
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Validación de Estado Fiel
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_validate_faithful_state_accepts_full_rank(self):
        """Debe aceptar estados con rango completo."""
        rho = create_quantum_mac_state(dimension=3, purity=0.7, seed=42)
        
        # No debe lanzar error
        TomitaTakesakiTheory.validate_faithful_state(rho.matrix)
    
    def test_validate_faithful_state_rejects_degenerate(self):
        """Debe rechazar estados con eigenvalores nulos."""
        # Estado degenerado: |0⟩⟨0|
        psi = np.array([1, 0, 0], dtype=np.complex128)
        rho = np.outer(psi, psi.conj())
        
        with pytest.raises(TraceAnomalyError, match="fiel"):
            TomitaTakesakiTheory.validate_faithful_state(rho)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Conjugación Modular
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_compute_modular_conjugation_full_rank_state(self):
        """Debe computar teoría modular para estado fiel."""
        rho = create_quantum_mac_state(dimension=3, purity=0.8, seed=42)
        
        modular_data = TomitaTakesakiTheory.compute_modular_conjugation(rho.matrix)
        
        assert isinstance(modular_data, ModularTheoryData)
        assert modular_data.modular_operator.shape == (3, 3)
        assert modular_data.modular_conjugation.shape == (3, 3)
        assert modular_data.is_faithful()
    
    def test_modular_operator_diagonal(self):
        """Operador modular debe ser diagonal en base espectral."""
        rho = create_quantum_mac_state(dimension=2, purity=0.9, seed=42)
        
        modular_data = TomitaTakesakiTheory.compute_modular_conjugation(rho.matrix)
        Delta = modular_data.modular_operator
        
        # Verificar que es diagonal
        off_diagonal = Delta - np.diag(np.diag(Delta))
        off_diagonal_norm = la.norm(off_diagonal, ord='fro')
        
        assert off_diagonal_norm < 1e-10, \
            f"Operador modular no diagonal: {off_diagonal_norm}"
    
    def test_modular_operator_positive_eigenvalues(self):
        """Operador modular debe tener eigenvalores positivos."""
        rho = create_quantum_mac_state(dimension=3, purity=0.7, seed=42)
        
        modular_data = TomitaTakesakiTheory.compute_modular_conjugation(rho.matrix)
        
        # Eigenvalues del operador modular
        eigenvalues = modular_data.eigenvalues
        
        assert np.all(eigenvalues > 0), \
            f"Eigenvalores no positivos: {eigenvalues}"
    
    def test_modular_conjugation_isometry(self):
        """Conjugación modular debe ser isometría."""
        rho = create_quantum_mac_state(dimension=2, purity=0.85, seed=42)
        
        modular_data = TomitaTakesakiTheory.compute_modular_conjugation(rho.matrix)
        J = modular_data.modular_conjugation
        
        # J debe satisfacer J J† = I
        J_Jdag = J @ J.conj().T
        identity = np.eye(2, dtype=np.complex128)
        
        error = la.norm(J_Jdag - identity, ord='fro')
        
        assert error < 1e-10, \
            f"Conjugación modular no es isometría: {error}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Relaciones de Tomita-Takesaki
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_verify_tomita_takesaki_J_involutive(self):
        """Debe verificar J² = I."""
        rho = create_quantum_mac_state(dimension=2, purity=0.9, seed=42)
        
        modular_data = TomitaTakesakiTheory.compute_modular_conjugation(rho.matrix)
        
        verification = TomitaTakesakiTheory.verify_tomita_takesaki_relations(
            modular_data
        )
        
        assert verification['J_involutive'], \
            f"J no involutivo: error={verification['identity_error']}"
    
    def test_verify_tomita_takesaki_J_conjugates_Delta(self):
        """Debe verificar J Δ J = Δ^{-1}."""
        rho = create_quantum_mac_state(dimension=2, purity=0.8, seed=42)
        
        modular_data = TomitaTakesakiTheory.compute_modular_conjugation(rho.matrix)
        
        verification = TomitaTakesakiTheory.verify_tomita_takesaki_relations(
            modular_data
        )
        
        assert verification['J_conjugates_Delta'], \
            f"J no conjuga Δ: error={verification['conjugation_error']}"
    
    def test_modular_automorphism_group_unitary(self):
        """Grupo modular debe ser unitario."""
        rho = create_quantum_mac_state(dimension=2, purity=0.75, seed=42)
        
        modular_data = TomitaTakesakiTheory.compute_modular_conjugation(rho.matrix)
        
        # Obtener automorfismo en t=1
        sigma_1 = modular_data.modular_automorphism_group(1.0)
        
        # Aplicar a un operador
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        X_evolved = sigma_1(X)
        
        # Verificar que preserva norma (unitariedad)
        norm_original = la.norm(X, ord='fro')
        norm_evolved = la.norm(X_evolved, ord='fro')
        
        assert np.isclose(norm_original, norm_evolved, atol=1e-10), \
            f"Automorfismo no preserva norma: {norm_original} vs {norm_evolved}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Clasificación de Factores
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_factor_classification_finite_dimension(self):
        """Estados de dimensión finita deben ser Tipo I_n."""
        rho = create_quantum_mac_state(dimension=3, purity=0.6, seed=42)
        
        modular_data = TomitaTakesakiTheory.compute_modular_conjugation(rho.matrix)
        
        assert modular_data.factor_type == VonNeumannFactorType.TYPE_I_FINITE, \
            f"Factor incorrecto: {modular_data.factor_type}"


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PYTEST Y FIXTURES GLOBALES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def setup_logging():
    """Configurar logging para pruebas."""
    logging.basicConfig(
        level=logging.WARNING,
        format='%(name)s - %(levelname)s - %(message)s'
    )


@pytest.fixture(scope="session")
def random_seed():
    """Semilla aleatoria global para reproducibilidad."""
    np.random.seed(42)
    return 42


# Marcadores personalizados
pytest.mark.cptp = pytest.mark.cptp
pytest.mark.composition = pytest.mark.composition
pytest.mark.orthomodular = pytest.mark.orthomodular
pytest.mark.tomita = pytest.mark.tomita
pytest.mark.algebra = pytest.mark.algebra
pytest.mark.slow = pytest.mark.slow


# ══════════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA PARA EJECUCIÓN DIRECTA
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--cov=app.wisdom.mac_algebra",
        "--cov-report=html",
        "--cov-report=term",
        "-m", "not slow",
        "--maxfail=10",
        "--durations=15"
    ])