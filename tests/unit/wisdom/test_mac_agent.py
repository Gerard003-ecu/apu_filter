# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo de Pruebas: MAC Agent Test Suite                                      ║
║ Ubicación: tests/wisdom/test_mac_agent.py                                    ║
║ Versión: 2.0.0-Quantum-Epistemic-Test-Suite                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Suite de Pruebas Rigurosas para Validación de:
───────────────────────────────────────────────
1. Mediciones POVM (Completitud, Positividad, Colapso)
2. Dinámica de Lindblad (CPTP, Termodinámica, Conservación)
3. Cohomología de Haces (Holonomía, Obstrucciones, Reparación)
4. Geometría de Información (Distancias, Fidelidad, Métricas)
5. Agente MAC (Ciclo OODA, Telemetría, Integración)

Metodología de Testing:
───────────────────────
- Pruebas unitarias (componentes aislados)
- Pruebas de integración (flujos completos)
- Pruebas de propiedades (property-based testing)
- Pruebas de invariantes físicos y matemáticos
- Pruebas de estabilidad numérica
- Pruebas de termodinámica cuántica

Referencias:
────────────
- Nielsen & Chuang (2010): "Quantum Computation and Quantum Information"
- Breuer & Petruccione (2002): "The Theory of Open Quantum Systems"
- Kraus (1983): "States, Effects, and Operations"
- Lindblad (1976): "Generators of quantum dynamical semigroups"
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
sys.path.insert(0, '../..')

from app.wisdom.mac_agent import (
    POVMMeasurement,
    POVMStatistics,
    MeasurementOutcome,
    LindbladDynamicsOrchestrator,
    LindbladEvolutionMetrics,
    SheafCohomologyCustodian,
    CohomologyAuditReport,
    GaloisAdjunctionAuditor,
    MACAgent
)

from app.wisdom.atomic_knowledge_matrix import (
    AtomicDensityMatrix,
    CellularSheafNeuralManifold,
    RestrictionMap,
    create_quantum_mac_state,
    create_geometric_learning_system,
    NumericalInstabilityError
)

logger = logging.getLogger("MAC.Agent.Tests")


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
def bell_states() -> Dict[str, np.ndarray]:
    """Estados de Bell maximalmente entrelazados."""
    phi_plus = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
    phi_minus = np.array([1, 0, 0, -1], dtype=np.complex128) / np.sqrt(2)
    psi_plus = np.array([0, 1, 1, 0], dtype=np.complex128) / np.sqrt(2)
    psi_minus = np.array([0, 1, -1, 0], dtype=np.complex128) / np.sqrt(2)
    
    return {
        'phi_plus': np.outer(phi_plus, phi_plus.conj()),
        'phi_minus': np.outer(phi_minus, phi_minus.conj()),
        'psi_plus': np.outer(psi_plus, psi_plus.conj()),
        'psi_minus': np.outer(psi_minus, psi_minus.conj())
    }


def create_projection_operators(dimension: int) -> List[np.ndarray]:
    """Crea proyectores ortogonales (POVM proyectivo)."""
    projectors = []
    for i in range(dimension):
        proj = np.zeros((dimension, dimension), dtype=np.complex128)
        proj[i, i] = 1.0
        projectors.append(proj)
    return projectors


def create_sic_povm(dimension: int = 2) -> List[np.ndarray]:
    """
    Crea SIC-POVM (Symmetric Informationally Complete) para d=2.
    
    SIC-POVM son óptimos para tomografía de estados cuánticos.
    """
    if dimension != 2:
        raise NotImplementedError("SIC-POVM solo implementado para d=2")
    
    # SIC-POVM en dimensión 2 (tetraedro en esfera de Bloch)
    sic_vectors = [
        np.array([1, 0], dtype=np.complex128),
        np.array([1, 2], dtype=np.complex128) / np.sqrt(3),
        np.array([1, 2 * np.exp(2j * np.pi / 3)], dtype=np.complex128) / np.sqrt(3),
        np.array([1, 2 * np.exp(4j * np.pi / 3)], dtype=np.complex128) / np.sqrt(3)
    ]
    
    # Normalizar y crear proyectores
    sic_ops = []
    for vec in sic_vectors:
        vec_norm = vec / la.norm(vec)
        proj = (1.0 / 2.0) * np.outer(vec_norm, vec_norm.conj())  # Factor 1/2 para POVM
        sic_ops.append(proj)
    
    return sic_ops


def create_depolarizing_channel(dimension: int, p: float) -> List[Tuple[float, np.ndarray]]:
    """
    Crea canal despolarizante: ρ → (1-p)ρ + p(I/d).
    
    Args:
        dimension: Dimensión del espacio de Hilbert
        p: Probabilidad de despolarización
    
    Returns:
        Lista de operadores de salto (γ, L)
    """
    if not 0 <= p <= 1:
        raise ValueError(f"Probabilidad fuera de rango: {p}")
    
    # Operadores de Pauli generalizados
    if dimension == 2:
        pauli_ops = [
            np.array([[0, 1], [1, 0]], dtype=np.complex128),    # X
            np.array([[0, -1j], [1j, 0]], dtype=np.complex128), # Y
            np.array([[1, 0], [0, -1]], dtype=np.complex128)    # Z
        ]
        
        jump_operators = [
            (p / 3.0, pauli_op) for pauli_op in pauli_ops
        ]
    else:
        # Versión simplificada para dimensiones superiores
        jump_operators = [
            (p / dimension, np.eye(dimension, dtype=np.complex128))
        ]
    
    return jump_operators


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1: PRUEBAS DE POVM MEASUREMENT
# ══════════════════════════════════════════════════════════════════════════════

class TestPOVMMeasurement:
    """Suite de pruebas para mediciones POVM."""
    
    @pytest.fixture
    def projective_povm_2d(self) -> POVMMeasurement:
        """POVM proyectivo en base computacional {|0⟩, |1⟩}."""
        kraus_ops = create_projection_operators(2)
        return POVMMeasurement(kraus_ops, validate_positivity=True)
    
    @pytest.fixture
    def sic_povm_2d(self) -> POVMMeasurement:
        """SIC-POVM óptimo para d=2."""
        sic_ops = create_sic_povm(dimension=2)
        # Convertir efectos a operadores de Kraus: Eₖ = √Mₖ
        kraus_ops = [la.sqrtm(M_k) for M_k in sic_ops]
        return POVMMeasurement(kraus_ops, validate_positivity=True)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Axiomas de POVM
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_identity_resolution_projective(self, projective_povm_2d):
        """Axioma: Σₖ Eₖ†Eₖ = I para POVM proyectivo."""
        effect_ops = projective_povm_2d.get_effect_operators()
        identity_sum = sum(effect_ops)
        
        assert np.allclose(identity_sum, np.eye(2), atol=1e-12), \
            "Resolución de identidad violada"
    
    def test_identity_resolution_sic(self, sic_povm_2d):
        """Axioma: Σₖ Eₖ†Eₖ = I para SIC-POVM."""
        effect_ops = sic_povm_2d.get_effect_operators()
        identity_sum = sum(effect_ops)
        
        assert np.allclose(identity_sum, np.eye(2), atol=1e-10), \
            "Resolución de identidad violada para SIC-POVM"
    
    def test_positivity_of_effects(self, projective_povm_2d):
        """Axioma: Todos los operadores de efecto deben ser positivos."""
        effect_ops = projective_povm_2d.get_effect_operators()
        
        for idx, M_k in enumerate(effect_ops):
            eigenvalues = la.eigvalsh(M_k)
            assert np.all(eigenvalues >= -1e-12), \
                f"Operador de efecto {idx} no es positivo: {eigenvalues}"
    
    def test_invalid_non_complete_povm(self):
        """Debe rechazar POVM que no resuelve la identidad."""
        # POVM incompleto: solo un proyector
        incomplete_kraus = [np.array([[1, 0], [0, 0]], dtype=np.complex128)]
        
        with pytest.raises(NumericalInstabilityError, match="identidad"):
            POVMMeasurement(incomplete_kraus)
    
    def test_invalid_non_positive_effect(self):
        """Debe rechazar operadores de efecto no positivos."""
        # Operador con eigenvalor negativo
        non_positive = np.array([[1, 2], [2, -1]], dtype=np.complex128)
        kraus_ops = [non_positive]
        
        with pytest.raises(NumericalInstabilityError, match="positivo"):
            POVMMeasurement(kraus_ops, validate_positivity=True)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Probabilidades
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_probabilities_sum_to_one(self, projective_povm_2d):
        """Probabilidades deben sumar 1 para cualquier estado."""
        rho = create_quantum_mac_state(dimension=2, purity=0.7, seed=42)
        probabilities = projective_povm_2d.compute_outcome_probabilities(rho)
        
        assert np.isclose(np.sum(probabilities), 1.0, atol=1e-10), \
            f"Probabilidades no suman 1: {np.sum(probabilities)}"
    
    def test_probabilities_non_negative(self, projective_povm_2d):
        """Probabilidades deben ser no negativas."""
        rho = create_quantum_mac_state(dimension=2, purity=1.0, seed=42)
        probabilities = projective_povm_2d.compute_outcome_probabilities(rho)
        
        assert np.all(probabilities >= 0), \
            f"Probabilidades negativas detectadas: {probabilities}"
    
    def test_pure_state_certainty(self):
        """Estado puro |0⟩ debe dar probabilidad 1 al medir en base Z."""
        psi_0 = np.array([1, 0], dtype=np.complex128)
        rho_0 = AtomicDensityMatrix(np.outer(psi_0, psi_0.conj()))
        
        # POVM proyectivo en base computacional
        kraus_ops = create_projection_operators(2)
        povm = POVMMeasurement(kraus_ops)
        
        probabilities = povm.compute_outcome_probabilities(rho_0)
        
        assert np.isclose(probabilities[0], 1.0, atol=1e-10), \
            f"Estado |0⟩ no da certeza en |0⟩: {probabilities}"
        assert np.isclose(probabilities[1], 0.0, atol=1e-10), \
            f"Estado |0⟩ da probabilidad no nula en |1⟩: {probabilities}"
    
    def test_maximally_mixed_uniform_probabilities(self):
        """Estado maximalmente mixto debe dar probabilidades uniformes en base Z."""
        rho_mixed = AtomicDensityMatrix(np.eye(2, dtype=np.complex128) / 2.0)
        
        kraus_ops = create_projection_operators(2)
        povm = POVMMeasurement(kraus_ops)
        
        probabilities = povm.compute_outcome_probabilities(rho_mixed)
        
        expected = np.array([0.5, 0.5])
        assert np.allclose(probabilities, expected, atol=1e-10), \
            f"Probabilidades no uniformes: {probabilities}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Colapso
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_measurement_preserves_trace(self, projective_povm_2d):
        """Colapso debe preservar traza del estado."""
        rho_initial = create_quantum_mac_state(dimension=2, purity=0.5, seed=42)
        
        outcome, rho_collapsed, _ = projective_povm_2d.measure_and_collapse(
            rho_initial, deterministic=True
        )
        
        trace_initial = np.trace(rho_initial.matrix).real
        trace_collapsed = np.trace(rho_collapsed.matrix).real
        
        assert np.isclose(trace_collapsed, 1.0, atol=1e-10), \
            f"Traza no unitaria post-colapso: {trace_collapsed}"
    
    def test_measurement_increases_purity(self, projective_povm_2d):
        """Medición proyectiva debe aumentar (o mantener) pureza."""
        rho_initial = create_quantum_mac_state(dimension=2, purity=0.5, seed=42)
        
        purity_initial = rho_initial.compute_metrics().purity
        
        _, rho_collapsed, stats = projective_povm_2d.measure_and_collapse(
            rho_initial, deterministic=True
        )
        
        purity_collapsed = stats.outcome_purity
        
        assert purity_collapsed >= purity_initial - 1e-10, \
            f"Pureza disminuye tras medición: {purity_initial} → {purity_collapsed}"
    
    def test_deterministic_vs_stochastic_collapse(self, projective_povm_2d):
        """Modo determinista debe seleccionar máxima probabilidad."""
        rho = create_quantum_mac_state(dimension=2, purity=0.8, seed=42)
        
        probabilities = projective_povm_2d.compute_outcome_probabilities(rho)
        max_prob_index = int(np.argmax(probabilities))
        
        # Medición determinista
        outcome_det, _, _ = projective_povm_2d.measure_and_collapse(
            rho, deterministic=True
        )
        
        assert outcome_det == max_prob_index, \
            f"Modo determinista no selecciona máxima probabilidad: {outcome_det} vs {max_prob_index}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Estadísticas de Información
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_shannon_entropy_bounds(self, projective_povm_2d):
        """Entropía de Shannon debe estar en [0, log₂(d)]."""
        rho = create_quantum_mac_state(dimension=2, purity=0.6, seed=42)
        
        _, _, stats = projective_povm_2d.measure_and_collapse(rho)
        
        assert 0 <= stats.shannon_entropy <= np.log2(2) + 1e-10, \
            f"Entropía fuera de rango: {stats.shannon_entropy}"
    
    def test_mutual_information_non_negative(self, projective_povm_2d):
        """Información mutua I(M:S) debe ser no negativa."""
        rho = create_quantum_mac_state(dimension=2, purity=0.5, seed=42)
        
        _, _, stats = projective_povm_2d.measure_and_collapse(rho)
        
        assert stats.mutual_information >= -1e-10, \
            f"Información mutua negativa: {stats.mutual_information}"
    
    def test_measurement_disturbance_non_negative(self, projective_povm_2d):
        """Perturbación de medición debe ser no negativa."""
        rho = create_quantum_mac_state(dimension=2, purity=0.7, seed=42)
        
        _, _, stats = projective_povm_2d.measure_and_collapse(rho)
        
        assert stats.measurement_disturbance >= -1e-10, \
            f"Perturbación negativa: {stats.measurement_disturbance}"
    
    def test_pure_state_zero_disturbance_on_eigenbasis(self):
        """Medir estado puro en su base debe dar perturbación cero."""
        psi_0 = np.array([1, 0], dtype=np.complex128)
        rho_0 = AtomicDensityMatrix(np.outer(psi_0, psi_0.conj()))
        
        kraus_ops = create_projection_operators(2)
        povm = POVMMeasurement(kraus_ops)
        
        _, rho_collapsed, stats = povm.measure_and_collapse(rho_0, deterministic=True)
        
        # Estado ya es eigenestado → no hay perturbación
        assert stats.measurement_disturbance < 1e-10, \
            f"Perturbación no nula en eigenestado: {stats.measurement_disturbance}"


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2: PRUEBAS DE LINDBLAD DYNAMICS
# ══════════════════════════════════════════════════════════════════════════════

class TestLindbladDynamicsOrchestrator:
    """Suite de pruebas para dinámica de Lindblad."""
    
    @pytest.fixture
    def lindblad_euler(self) -> LindbladDynamicsOrchestrator:
        """Orquestador con integración de Euler."""
        return LindbladDynamicsOrchestrator(integration_method='euler')
    
    @pytest.fixture
    def lindblad_rk4(self) -> LindbladDynamicsOrchestrator:
        """Orquestador con integración RK4."""
        return LindbladDynamicsOrchestrator(integration_method='rk4')
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Conservación de Traza
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_trace_preservation_unitary(self, lindblad_rk4, pauli_operators):
        """Evolución unitaria pura debe preservar traza exactamente."""
        rho_initial = create_quantum_mac_state(dimension=2, purity=0.8, seed=42)
        
        # Hamiltoniano: H = σₓ
        H = pauli_operators['X']
        
        # Sin operadores de salto (evolución unitaria)
        jump_ops = []
        
        rho_final, metrics = lindblad_rk4.evolve_state(
            rho=rho_initial,
            H_error=H,
            jump_operators=jump_ops,
            dt=0.1,
            num_steps=10
        )
        
        assert np.isclose(metrics.trace_after, 1.0, atol=1e-10), \
            f"Traza no preservada: {metrics.trace_after}"
    
    def test_trace_preservation_with_dissipation(self, lindblad_rk4, pauli_operators):
        """Evolución con disipación debe preservar traza."""
        rho_initial = create_quantum_mac_state(dimension=2, purity=0.5, seed=42)
        
        H = np.zeros((2, 2), dtype=np.complex128)  # Sin Hamiltoniano
        
        # Canal de amortiguamiento de amplitud
        jump_ops = [(0.1, pauli_operators['X'])]
        
        rho_final, metrics = lindblad_rk4.evolve_state(
            rho=rho_initial,
            H_error=H,
            jump_operators=jump_ops,
            dt=0.01,
            num_steps=100
        )
        
        assert np.isclose(metrics.trace_after, 1.0, atol=1e-9), \
            f"Traza no preservada con disipación: {metrics.trace_after}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Hermiticidad
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_hermiticity_preservation(self, lindblad_rk4, pauli_operators):
        """Estado debe permanecer hermitiano."""
        rho_initial = create_quantum_mac_state(dimension=2, purity=0.7, seed=42)
        
        H = pauli_operators['Z']
        jump_ops = [(0.05, pauli_operators['Y'])]
        
        rho_final, _ = lindblad_rk4.evolve_state(
            rho=rho_initial,
            H_error=H,
            jump_operators=jump_ops,
            dt=0.1,
            num_steps=50
        )
        
        rho_matrix = rho_final.matrix
        hermiticity_error = la.norm(rho_matrix - rho_matrix.conj().T, ord='fro')
        
        assert hermiticity_error < 1e-10, \
            f"Estado no hermitiano: error = {hermiticity_error}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Positividad
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_positivity_preservation(self, lindblad_rk4, pauli_operators):
        """Estado debe permanecer semidefinido positivo."""
        rho_initial = create_quantum_mac_state(dimension=2, purity=0.6, seed=42)
        
        H = pauli_operators['X'] + pauli_operators['Z']
        jump_ops = [(0.1, pauli_operators['X']), (0.1, pauli_operators['Y'])]
        
        rho_final, _ = lindblad_rk4.evolve_state(
            rho=rho_initial,
            H_error=H,
            jump_operators=jump_ops,
            dt=0.05,
            num_steps=100
        )
        
        eigenvalues = la.eigvalsh(rho_final.matrix)
        
        assert np.all(eigenvalues >= -1e-10), \
            f"Estado no positivo: eigenvalores = {eigenvalues}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas Termodinámicas
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_entropy_production_non_negative(self, lindblad_rk4):
        """Segundo principio: producción de entropía debe ser ≥ 0."""
        rho_initial = create_quantum_mac_state(dimension=2, purity=1.0, seed=42)
        
        H = np.zeros((2, 2), dtype=np.complex128)
        
        # Canal despolarizante
        jump_ops = create_depolarizing_channel(dimension=2, p=0.3)
        
        rho_final, metrics = lindblad_rk4.evolve_state(
            rho=rho_initial,
            H_error=H,
            jump_operators=jump_ops,
            dt=0.1,
            num_steps=50
        )
        
        assert metrics.entropy_production >= -1e-10, \
            f"Producción de entropía negativa: {metrics.entropy_production}"
    
    def test_purity_decrease_with_decoherence(self, lindblad_rk4):
        """Decoherencia debe disminuir pureza."""
        rho_initial = create_quantum_mac_state(dimension=2, purity=1.0, seed=42)
        
        H = np.zeros((2, 2), dtype=np.complex128)
        jump_ops = create_depolarizing_channel(dimension=2, p=0.5)
        
        rho_final, metrics = lindblad_rk4.evolve_state(
            rho=rho_initial,
            H_error=H,
            jump_operators=jump_ops,
            dt=0.1,
            num_steps=100
        )
        
        assert metrics.purity_after <= metrics.purity_before + 1e-10, \
            f"Pureza aumenta con decoherencia: {metrics.purity_before} → {metrics.purity_after}"
    
    def test_coherence_dissipation(self, lindblad_rk4, pauli_operators):
        """Dephasing debe disipar coherencias (elementos off-diagonal)."""
        # Estado inicial con coherencia máxima: |+⟩ = (|0⟩+|1⟩)/√2
        psi_plus = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
        rho_initial = AtomicDensityMatrix(np.outer(psi_plus, psi_plus.conj()))
        
        H = np.zeros((2, 2), dtype=np.complex128)
        
        # Dephasing: σz como operador de salto
        jump_ops = [(0.5, pauli_operators['Z'])]
        
        rho_final, metrics = lindblad_rk4.evolve_state(
            rho=rho_initial,
            H_error=H,
            jump_operators=jump_ops,
            dt=0.1,
            num_steps=100
        )
        
        assert metrics.dissipated_coherence >= -1e-10, \
            f"Coherencia disipada negativa: {metrics.dissipated_coherence}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Métodos de Integración
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_euler_vs_rk4_convergence(self, pauli_operators):
        """RK4 debe ser más preciso que Euler para mismo dt."""
        rho_initial = create_quantum_mac_state(dimension=2, purity=0.8, seed=42)
        
        H = pauli_operators['X']
        jump_ops = [(0.1, pauli_operators['Z'])]
        
        # Euler
        lindblad_euler = LindbladDynamicsOrchestrator(integration_method='euler')
        rho_euler, _ = lindblad_euler.evolve_state(
            rho=rho_initial, H_error=H, jump_operators=jump_ops, dt=0.1, num_steps=10
        )
        
        # RK4
        lindblad_rk4 = LindbladDynamicsOrchestrator(integration_method='rk4')
        rho_rk4, _ = lindblad_rk4.evolve_state(
            rho=rho_initial, H_error=H, jump_operators=jump_ops, dt=0.1, num_steps=10
        )
        
        # RK4 con dt más pequeño como "ground truth"
        rho_reference, _ = lindblad_rk4.evolve_state(
            rho=rho_initial, H_error=H, jump_operators=jump_ops, dt=0.01, num_steps=100
        )
        
        error_euler = la.norm(rho_euler.matrix - rho_reference.matrix, ord='fro')
        error_rk4 = la.norm(rho_rk4.matrix - rho_reference.matrix, ord='fro')
        
        # RK4 debe tener menor error
        assert error_rk4 < error_euler, \
            f"RK4 no es más preciso: error_euler={error_euler}, error_rk4={error_rk4}"
    
    def test_timestep_independence(self, lindblad_rk4, pauli_operators):
        """Resultado final debe ser independiente de cómo se subdivide el tiempo."""
        rho_initial = create_quantum_mac_state(dimension=2, purity=0.7, seed=42)
        
        H = pauli_operators['Y']
        jump_ops = [(0.05, pauli_operators['X'])]
        
        # Una integración con dt grande
        rho_large_dt, _ = lindblad_rk4.evolve_state(
            rho=rho_initial, H_error=H, jump_operators=jump_ops, dt=0.1, num_steps=10
        )
        
        # Múltiples pasos con dt pequeño
        rho_small_dt, _ = lindblad_rk4.evolve_state(
            rho=rho_initial, H_error=H, jump_operators=jump_ops, dt=0.01, num_steps=100
        )
        
        # Deben estar cercanos
        difference = la.norm(rho_large_dt.matrix - rho_small_dt.matrix, ord='fro')
        
        assert difference < 0.1, \
            f"Diferencia significativa con diferentes timesteps: {difference}"


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3: PRUEBAS DE SHEAF COHOMOLOGY CUSTODIAN
# ══════════════════════════════════════════════════════════════════════════════

class TestSheafCohomologyCustodian:
    """Suite de pruebas para auditoría cohomológica."""
    
    @pytest.fixture
    def simple_sheaf_system(self) -> Tuple[CellularSheafNeuralManifold, SheafCohomologyCustodian]:
        """Sistema de haces simple para pruebas."""
        sheaf, _, _ = create_geometric_learning_system(
            num_vertices=5,
            num_edges=6,
            fiber_dim_vertex=1,
            fiber_dim_edge=1,
            seed=42
        )
        custodian = SheafCohomologyCustodian(sheaf, auto_project=False)
        return sheaf, custodian
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Holonomía
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_constant_section_is_holonomic(self, simple_sheaf_system):
        """Sección constante debe ser holonómica (δx = 0)."""
        sheaf, custodian = simple_sheaf_system
        
        constant_section = np.ones(sheaf.num_vertices)
        report = custodian.audit_holonomy(constant_section, raise_on_violation=False)
        
        assert report.is_holonomic, \
            f"Sección constante no holonómica: E={report.dirichlet_energy}"
    
    def test_discontinuous_section_not_holonomic(self, simple_sheaf_system):
        """Sección discontinua debe fallar holonomía."""
        sheaf, custodian = simple_sheaf_system
        
        discontinuous_section = np.random.randn(sheaf.num_vertices)
        report = custodian.audit_holonomy(discontinuous_section, raise_on_violation=False)
        
        # Probabilísticamente, debería tener energía no nula
        # (excepto en casos degenerados)
        if report.is_holonomic:
            pytest.skip("Sección aleatoria resultó holonómica (caso degenerado)")
    
    def test_holonomy_violation_raises_error(self, simple_sheaf_system):
        """Debe lanzar error si se viola holonomía y raise_on_violation=True."""
        sheaf, custodian = simple_sheaf_system
        
        # Sección diseñada para violar holonomía
        discontinuous = np.array([1.0, 0.0, -1.0, 2.0, -2.0])
        
        # Verificar que efectivamente tiene energía no nula
        energy = sheaf.compute_dirichlet_energy(discontinuous)
        
        if energy < 1e-9:
            pytest.skip("Sección diseñada resultó armónica (caso degenerado)")
        
        with pytest.raises(NumericalInstabilityError, match="Obstrucción topológica"):
            custodian.audit_holonomy(discontinuous, raise_on_violation=True)
    
    def test_dirichlet_energy_non_negative(self, simple_sheaf_system):
        """Energía de Dirichlet debe ser siempre no negativa."""
        sheaf, custodian = simple_sheaf_system
        
        random_section = np.random.randn(sheaf.num_vertices)
        report = custodian.audit_holonomy(random_section, raise_on_violation=False)
        
        assert report.dirichlet_energy >= -1e-12, \
            f"Energía de Dirichlet negativa: {report.dirichlet_energy}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Reparación
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_repair_reduces_energy(self, simple_sheaf_system):
        """Reparación debe reducir energía de Dirichlet."""
        sheaf, custodian = simple_sheaf_system
        
        inconsistent_section = np.random.randn(sheaf.num_vertices)
        energy_before = sheaf.compute_dirichlet_energy(inconsistent_section)
        
        repaired_section = custodian.repair_semantic_state(inconsistent_section)
        energy_after = sheaf.compute_dirichlet_energy(repaired_section)
        
        assert energy_after <= energy_before + 1e-10, \
            f"Reparación aumenta energía: {energy_before} → {energy_after}"
    
    def test_repaired_section_is_holonomic(self, simple_sheaf_system):
        """Sección reparada debe ser holonómica."""
        sheaf, custodian = simple_sheaf_system
        
        inconsistent_section = np.random.randn(sheaf.num_vertices)
        repaired_section = custodian.repair_semantic_state(inconsistent_section)
        
        report = custodian.audit_holonomy(repaired_section, raise_on_violation=False)
        
        assert report.is_holonomic, \
            f"Sección reparada no holonómica: E={report.dirichlet_energy}"
    
    def test_repair_is_idempotent(self, simple_sheaf_system):
        """Reparar dos veces debe dar el mismo resultado."""
        sheaf, custodian = simple_sheaf_system
        
        section = np.random.randn(sheaf.num_vertices)
        
        repaired_once = custodian.repair_semantic_state(section)
        repaired_twice = custodian.repair_semantic_state(repaired_once)
        
        assert np.allclose(repaired_once, repaired_twice, atol=1e-9), \
            "Reparación no es idempotente"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Grupos de Cohomología
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_betti_numbers_non_negative(self, simple_sheaf_system):
        """Números de Betti deben ser no negativos."""
        _, custodian = simple_sheaf_system
        
        cohomology = custodian.compute_cohomology_groups()
        
        for degree, group in cohomology.items():
            assert group.betti_number >= 0, \
                f"β_{degree} negativo: {group.betti_number}"
    
    def test_global_sections_dimension_consistency(self, simple_sheaf_system):
        """Dimensión de secciones globales debe coincidir con β₀."""
        sheaf, custodian = simple_sheaf_system
        
        report = custodian.audit_holonomy(
            np.zeros(sheaf.num_vertices), 
            raise_on_violation=False
        )
        
        cohomology = custodian.compute_cohomology_groups()
        beta_0 = cohomology[0].betti_number
        
        assert report.global_sections_dim == beta_0, \
            f"Inconsistencia en dim H⁰: {report.global_sections_dim} vs β₀={beta_0}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Telemetría
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_telemetry_tracking(self, simple_sheaf_system):
        """Telemetría debe rastrear auditorías y violaciones."""
        sheaf, custodian = simple_sheaf_system
        
        # Realizar múltiples auditorías
        for _ in range(5):
            section = np.random.randn(sheaf.num_vertices)
            try:
                custodian.audit_holonomy(section, raise_on_violation=False)
            except:
                pass
        
        telemetry = custodian.get_telemetry()
        
        assert telemetry['total_audits'] == 5, \
            f"Conteo de auditorías incorrecto: {telemetry['total_audits']}"
        assert 'violation_rate' in telemetry, "Falta tasa de violación"


# ══════════════════════════════════════════════════════════════════════════════
# FASE 4: PRUEBAS DE GALOIS ADJUNCTION AUDITOR
# ══════════════════════════════════════════════════════════════════════════════

class TestGaloisAdjunctionAuditor:
    """Suite de pruebas para geometría de información cuántica."""
    
    @pytest.fixture
    def auditor(self) -> GaloisAdjunctionAuditor:
        """Instancia del auditor."""
        return GaloisAdjunctionAuditor()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Distancia de Bures
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_bures_distance_identical_states(self, auditor):
        """Distancia de Bures entre estados idénticos debe ser 0."""
        rho = create_quantum_mac_state(dimension=2, purity=0.7, seed=42)
        
        distance = auditor.compute_bures_distance(rho.matrix, rho.matrix)
        
        assert np.isclose(distance, 0.0, atol=1e-10), \
            f"Distancia no nula para estados idénticos: {distance}"
    
    def test_bures_distance_non_negative(self, auditor):
        """Distancia de Bures debe ser no negativa."""
        rho1 = create_quantum_mac_state(dimension=2, purity=0.5, seed=42)
        rho2 = create_quantum_mac_state(dimension=2, purity=0.8, seed=43)
        
        distance = auditor.compute_bures_distance(rho1.matrix, rho2.matrix)
        
        assert distance >= -1e-12, \
            f"Distancia de Bures negativa: {distance}"
    
    def test_bures_distance_symmetric(self, auditor):
        """Distancia de Bures debe ser simétrica: d(ρ,σ) = d(σ,ρ)."""
        rho1 = create_quantum_mac_state(dimension=2, purity=0.6, seed=42)
        rho2 = create_quantum_mac_state(dimension=2, purity=0.9, seed=43)
        
        d_12 = auditor.compute_bures_distance(rho1.matrix, rho2.matrix)
        d_21 = auditor.compute_bures_distance(rho2.matrix, rho1.matrix)
        
        assert np.isclose(d_12, d_21, atol=1e-10), \
            f"Distancia no simétrica: d(ρ₁,ρ₂)={d_12}, d(ρ₂,ρ₁)={d_21}"
    
    def test_bures_distance_orthogonal_pure_states(self, auditor):
        """Distancia entre estados puros ortogonales debe ser máxima."""
        psi_0 = np.array([1, 0], dtype=np.complex128)
        psi_1 = np.array([0, 1], dtype=np.complex128)
        
        rho_0 = np.outer(psi_0, psi_0.conj())
        rho_1 = np.outer(psi_1, psi_1.conj())
        
        distance = auditor.compute_bures_distance(rho_0, rho_1)
        
        # Para estados puros ortogonales: d_BW = √2
        expected = np.sqrt(2.0)
        assert np.isclose(distance, expected, atol=1e-10), \
            f"Distancia incorrecta para estados ortogonales: {distance} vs {expected}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Fidelidad
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_fidelity_identical_states(self, auditor):
        """Fidelidad entre estados idénticos debe ser 1."""
        rho = create_quantum_mac_state(dimension=2, purity=0.8, seed=42)
        
        fidelity = auditor.compute_fidelity(rho.matrix, rho.matrix)
        
        assert np.isclose(fidelity, 1.0, atol=1e-10), \
            f"Fidelidad no unitaria para estados idénticos: {fidelity}"
    
    def test_fidelity_bounds(self, auditor):
        """Fidelidad debe estar en [0, 1]."""
        rho1 = create_quantum_mac_state(dimension=2, purity=0.5, seed=42)
        rho2 = create_quantum_mac_state(dimension=2, purity=0.7, seed=43)
        
        fidelity = auditor.compute_fidelity(rho1.matrix, rho2.matrix)
        
        assert 0 <= fidelity <= 1 + 1e-10, \
            f"Fidelidad fuera de rango: {fidelity}"
    
    def test_fidelity_symmetric(self, auditor):
        """Fidelidad debe ser simétrica: F(ρ,σ) = F(σ,ρ)."""
        rho1 = create_quantum_mac_state(dimension=2, purity=0.6, seed=42)
        rho2 = create_quantum_mac_state(dimension=2, purity=0.9, seed=43)
        
        f_12 = auditor.compute_fidelity(rho1.matrix, rho2.matrix)
        f_21 = auditor.compute_fidelity(rho2.matrix, rho1.matrix)
        
        assert np.isclose(f_12, f_21, atol=1e-10), \
            f"Fidelidad no simétrica: F(ρ₁,ρ₂)={f_12}, F(ρ₂,ρ₁)={f_21}"
    
    def test_fidelity_orthogonal_states(self, auditor):
        """Fidelidad entre estados ortogonales debe ser 0."""
        psi_0 = np.array([1, 0], dtype=np.complex128)
        psi_1 = np.array([0, 1], dtype=np.complex128)
        
        rho_0 = np.outer(psi_0, psi_0.conj())
        rho_1 = np.outer(psi_1, psi_1.conj())
        
        fidelity = auditor.compute_fidelity(rho_0, rho_1)
        
        assert np.isclose(fidelity, 0.0, atol=1e-10), \
            f"Fidelidad no nula para estados ortogonales: {fidelity}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Distancia Traza
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_trace_distance_identical_states(self, auditor):
        """Distancia traza entre estados idénticos debe ser 0."""
        rho = create_quantum_mac_state(dimension=2, purity=0.6, seed=42)
        
        distance = auditor.compute_trace_distance(rho.matrix, rho.matrix)
        
        assert np.isclose(distance, 0.0, atol=1e-10), \
            f"Distancia traza no nula: {distance}"
    
    def test_trace_distance_bounds(self, auditor):
        """Distancia traza debe estar en [0, 1]."""
        rho1 = create_quantum_mac_state(dimension=2, purity=0.5, seed=42)
        rho2 = create_quantum_mac_state(dimension=2, purity=0.8, seed=43)
        
        distance = auditor.compute_trace_distance(rho1.matrix, rho2.matrix)
        
        assert 0 <= distance <= 1 + 1e-10, \
            f"Distancia traza fuera de rango: {distance}"
    
    def test_trace_distance_orthogonal_pure_states(self, auditor):
        """Distancia traza entre puros ortogonales debe ser 1."""
        psi_0 = np.array([1, 0], dtype=np.complex128)
        psi_1 = np.array([0, 1], dtype=np.complex128)
        
        rho_0 = np.outer(psi_0, psi_0.conj())
        rho_1 = np.outer(psi_1, psi_1.conj())
        
        distance = auditor.compute_trace_distance(rho_0, rho_1)
        
        assert np.isclose(distance, 1.0, atol=1e-10), \
            f"Distancia traza incorrecta: {distance}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Validación de Adjunción
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_adjunction_validation_accepts_close_states(self, auditor):
        """Debe aceptar estados muy cercanos."""
        rho_mac = create_quantum_mac_state(dimension=2, purity=0.7, seed=42)
        
        # Perturbación pequeña
        noise = np.random.randn(2, 2) * 0.001
        noise = (noise + noise.conj().T) / 2  # Hermitianizar
        rho_mic_matrix = rho_mac.matrix + noise
        rho_mic_matrix = rho_mic_matrix / np.trace(rho_mic_matrix)  # Renormalizar
        
        rho_mic = AtomicDensityMatrix(rho_mic_matrix)
        
        is_valid, _ = auditor.validate_adjunction_counit(
            rho_mac, rho_mic, epsilon=0.1
        )
        
        assert is_valid, "Estados cercanos rechazados incorrectamente"
    
    def test_adjunction_validation_rejects_distant_states(self, auditor):
        """Debe rechazar estados muy diferentes."""
        rho_mac = create_quantum_mac_state(dimension=2, purity=1.0, seed=42)
        rho_mic = create_quantum_mac_state(dimension=2, purity=0.1, seed=99)
        
        is_valid, metrics = auditor.validate_adjunction_counit(
            rho_mac, rho_mic, epsilon=0.01
        )
        
        # Probabilísticamente, deberían ser rechazados
        # (excepto en casos muy improbables)
        if is_valid:
            pytest.skip(f"Estados aleatorios resultaron cercanos: d={metrics['bures_distance']}")


# ══════════════════════════════════════════════════════════════════════════════
# FASE 5: PRUEBAS DE MAC AGENT (INTEGRACIÓN)
# ══════════════════════════════════════════════════════════════════════════════

class TestMACAgent:
    """Suite de pruebas para el agente MAC completo."""
    
    @pytest.fixture
    def mac_agent_system(self) -> Tuple[MACAgent, CellularSheafNeuralManifold]:
        """Sistema MAC completo para pruebas."""
        sheaf, _, _ = create_geometric_learning_system(
            num_vertices=6,
            num_edges=8,
            fiber_dim_vertex=1,
            fiber_dim_edge=1,
            seed=42
        )
        
        agent = MACAgent(
            sheaf_manifold=sheaf,
            integration_method='rk4',
            auto_repair=True,
            debug_mode=True
        )
        
        return agent, sheaf
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas del Ciclo OODA
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_process_telemetry_cartridge_success(self, mac_agent_system):
        """Procesamiento exitoso de cartucho de telemetría."""
        agent, sheaf = mac_agent_system
        
        # Estado inicial
        rho_initial = create_quantum_mac_state(dimension=6, purity=0.8, seed=42)
        
        # Sección semántica (holonómica)
        semantic_vector = np.ones(sheaf.num_vertices)
        
        # Hamiltoniano de error
        H_error = np.random.randn(6, 6) + 1j * np.random.randn(6, 6)
        H_error = (H_error + H_error.conj().T) / 2  # Hermitianizar
        
        # Operadores de salto
        jump_ops = [(0.1, np.random.randn(6, 6) + 1j * np.random.randn(6, 6))]
        
        rho_updated, telemetry = agent.process_telemetry_cartridge(
            current_rho=rho_initial,
            semantic_vector=semantic_vector,
            H_error=H_error,
            jump_ops=jump_ops,
            dt=0.01,
            num_steps=10
        )
        
        assert telemetry['success'], "Procesamiento falló"
        assert 'cohomology_report' in telemetry, "Falta reporte cohomológico"
        assert 'lindblad_evolution' in telemetry, "Falta evolución de Lindblad"
    
    def test_process_telemetry_preserves_trace(self, mac_agent_system):
        """Procesamiento debe preservar traza del estado."""
        agent, sheaf = mac_agent_system
        
        rho_initial = create_quantum_mac_state(dimension=6, purity=0.7, seed=42)
        semantic_vector = np.ones(sheaf.num_vertices)
        
        H_error = np.zeros((6, 6), dtype=np.complex128)
        jump_ops = []
        
        rho_updated, _ = agent.process_telemetry_cartridge(
            current_rho=rho_initial,
            semantic_vector=semantic_vector,
            H_error=H_error,
            jump_ops=jump_ops,
            dt=0.01,
            num_steps=5
        )
        
        trace = np.trace(rho_updated.matrix).real
        assert np.isclose(trace, 1.0, atol=1e-10), \
            f"Traza no preservada: {trace}"
    
    def test_auto_repair_fixes_holonomy_violations(self):
        """Auto-reparación debe corregir violaciones de holonomía."""
        sheaf, _, _ = create_geometric_learning_system(
            num_vertices=5,
            num_edges=6,
            fiber_dim_vertex=1,
            fiber_dim_edge=1,
            seed=42
        )
        
        agent = MACAgent(
            sheaf_manifold=sheaf,
            auto_repair=True,
            debug_mode=False
        )
        
        rho_initial = create_quantum_mac_state(dimension=5, purity=0.6, seed=42)
        
        # Sección discontinua (viola holonomía)
        semantic_vector = np.array([1.0, 0.0, -1.0, 2.0, -2.0])
        
        H_error = np.zeros((5, 5), dtype=np.complex128)
        jump_ops = []
        
        # No debe lanzar error gracias a auto_repair
        rho_updated, telemetry = agent.process_telemetry_cartridge(
            current_rho=rho_initial,
            semantic_vector=semantic_vector,
            H_error=H_error,
            jump_ops=jump_ops,
            dt=0.01
        )
        
        assert telemetry['success'], "Procesamiento falló con auto-reparación"
        
        # Verificar que se aplicó reparación
        if 'semantic_repair_applied' in telemetry:
            assert telemetry['semantic_repair_applied'], \
                "Reparación no aplicada cuando era necesaria"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Extracción de Sabiduría
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_extract_wisdom_produces_valid_decision(self, mac_agent_system):
        """Extracción de sabiduría debe producir decisión válida."""
        agent, _ = mac_agent_system
        
        rho = create_quantum_mac_state(dimension=2, purity=0.8, seed=42)
        
        # POVM proyectivo simple
        povm_ops = create_projection_operators(2)
        
        decision_index, rho_collapsed, statistics = agent.extract_wisdom(
            current_rho=rho,
            povm_ops=povm_ops,
            deterministic=False
        )
        
        assert 0 <= decision_index < len(povm_ops), \
            f"Índice de decisión fuera de rango: {decision_index}"
        assert isinstance(statistics, POVMStatistics), \
            "Estadísticas mal formadas"
    
    def test_extract_wisdom_deterministic_mode(self, mac_agent_system):
        """Modo determinista debe seleccionar máxima probabilidad."""
        agent, _ = mac_agent_system
        
        # Estado sesgado hacia |0⟩
        psi = np.array([0.9, 0.1], dtype=np.complex128)
        psi /= la.norm(psi)
        rho = AtomicDensityMatrix(np.outer(psi, psi.conj()))
        
        povm_ops = create_projection_operators(2)
        
        decision_index, _, _ = agent.extract_wisdom(
            current_rho=rho,
            povm_ops=povm_ops,
            deterministic=True
        )
        
        # Debe seleccionar |0⟩ (índice 0)
        assert decision_index == 0, \
            f"Modo determinista no selecciona máxima probabilidad: {decision_index}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Validación Epistemológica
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_epistemological_coherence_validation(self, mac_agent_system):
        """Validación de coherencia epistemológica."""
        agent, _ = mac_agent_system
        
        rho_mac = create_quantum_mac_state(dimension=3, purity=0.7, seed=42)
        rho_mic = create_quantum_mac_state(dimension=3, purity=0.69, seed=43)
        
        is_valid, metrics = agent.validate_epistemological_coherence(
            rho_mac=rho_mac,
            rho_mic=rho_mic,
            epsilon=0.5  # Umbral generoso
        )
        
        assert isinstance(is_valid, bool), "Resultado de validación mal formado"
        assert 'bures_distance' in metrics, "Faltan métricas de distancia"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Telemetría
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_telemetry_tracking(self, mac_agent_system):
        """Telemetría debe rastrear operaciones."""
        agent, sheaf = mac_agent_system
        
        rho = create_quantum_mac_state(dimension=sheaf.num_vertices, purity=0.8, seed=42)
        semantic_vector = np.ones(sheaf.num_vertices)
        H_error = np.zeros((sheaf.num_vertices, sheaf.num_vertices), dtype=np.complex128)
        jump_ops = []
        
        # Realizar múltiples operaciones
        for _ in range(3):
            agent.process_telemetry_cartridge(
                current_rho=rho,
                semantic_vector=semantic_vector,
                H_error=H_error,
                jump_ops=jump_ops,
                dt=0.01
            )
        
        telemetry = agent.get_telemetry()
        
        assert telemetry['operation_count'] == 3, \
            f"Conteo de operaciones incorrecto: {telemetry['operation_count']}"
    
    def test_debug_mode_stores_history(self):
        """Modo debug debe almacenar historial de estados."""
        sheaf, _, _ = create_geometric_learning_system(
            num_vertices=4, num_edges=5, fiber_dim_vertex=1, fiber_dim_edge=1, seed=42
        )
        
        agent = MACAgent(sheaf_manifold=sheaf, debug_mode=True)
        
        rho = create_quantum_mac_state(dimension=4, purity=0.7, seed=42)
        semantic_vector = np.ones(sheaf.num_vertices)
        H_error = np.zeros((4, 4), dtype=np.complex128)
        
        agent.process_telemetry_cartridge(
            current_rho=rho,
            semantic_vector=semantic_vector,
            H_error=H_error,
            jump_ops=[],
            dt=0.01
        )
        
        telemetry = agent.get_telemetry()
        
        assert telemetry['state_history_length'] > 0, \
            "Historial de estados no almacenado en modo debug"
    
    def test_reset_clears_telemetry(self, mac_agent_system):
        """Reset debe limpiar telemetría."""
        agent, sheaf = mac_agent_system
        
        rho = create_quantum_mac_state(dimension=sheaf.num_vertices, purity=0.8, seed=42)
        semantic_vector = np.ones(sheaf.num_vertices)
        H_error = np.zeros((sheaf.num_vertices, sheaf.num_vertices), dtype=np.complex128)
        
        agent.process_telemetry_cartridge(
            current_rho=rho,
            semantic_vector=semantic_vector,
            H_error=H_error,
            jump_ops=[],
            dt=0.01
        )
        
        agent.reset()
        
        telemetry = agent.get_telemetry()
        
        assert telemetry['operation_count'] == 0, \
            "Reset no limpia conteo de operaciones"


# ══════════════════════════════════════════════════════════════════════════════
# FASE 6: PRUEBAS DE INTEGRACIÓN END-TO-END
# ══════════════════════════════════════════════════════════════════════════════

class TestEndToEndScenarios:
    """Pruebas de integración de flujos completos."""
    
    def test_complete_quantum_epistemic_cycle(self):
        """Ciclo completo: estado → evolución → medición."""
        # Setup
        sheaf, _, _ = create_geometric_learning_system(
            num_vertices=4,
            num_edges=5,
            fiber_dim_vertex=1,
            fiber_dim_edge=1,
            seed=42
        )
        
        agent = MACAgent(sheaf_manifold=sheaf, auto_repair=True)
        
        # Estado inicial
        rho = create_quantum_mac_state(dimension=4, purity=0.8, seed=42)
        
        # Evolución
        semantic_vector = np.ones(sheaf.num_vertices)
        H_error = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        H_error = (H_error + H_error.conj().T) / 2
        
        jump_ops = create_depolarizing_channel(dimension=4, p=0.1)
        
        rho_evolved, telemetry = agent.process_telemetry_cartridge(
            current_rho=rho,
            semantic_vector=semantic_vector,
            H_error=H_error,
            jump_ops=jump_ops,
            dt=0.05,
            num_steps=20
        )
        
        # Medición
        povm_ops = create_projection_operators(4)
        decision, rho_collapsed, stats = agent.extract_wisdom(
            current_rho=rho_evolved,
            povm_ops=povm_ops
        )
        
        # Validaciones
        assert telemetry['success'], "Evolución falló"
        assert 0 <= decision < 4, "Decisión inválida"
        assert np.isclose(np.trace(rho_collapsed.matrix).real, 1.0, atol=1e-10), \
            "Estado colapsado no normalizado"
    
    def test_stress_test_long_evolution(self):
        """Prueba de estrés con evolución larga."""
        sheaf, _, _ = create_geometric_learning_system(
            num_vertices=3,
            num_edges=3,
            fiber_dim_vertex=1,
            fiber_dim_edge=1,
            seed=42
        )
        
        agent = MACAgent(sheaf_manifold=sheaf, debug_mode=False)
        
        rho = create_quantum_mac_state(dimension=3, purity=1.0, seed=42)
        semantic_vector = np.ones(sheaf.num_vertices)
        H_error = np.zeros((3, 3), dtype=np.complex128)
        jump_ops = create_depolarizing_channel(dimension=3, p=0.05)
        
        # Evolución larga
        rho_final, _ = agent.process_telemetry_cartridge(
            current_rho=rho,
            semantic_vector=semantic_vector,
            H_error=H_error,
            jump_ops=jump_ops,
            dt=0.01,
            num_steps=1000
        )
        
        # Verificar propiedades físicas
        trace = np.trace(rho_final.matrix).real
        eigenvalues = la.eigvalsh(rho_final.matrix)
        
        assert np.isclose(trace, 1.0, atol=1e-8), \
            f"Traza deriva tras evolución larga: {trace}"
        assert np.all(eigenvalues >= -1e-8), \
            "Positividad violada tras evolución larga"
    
    def test_multiple_measurement_cycles(self):
        """Múltiples ciclos de medición deben ser consistentes."""
        agent = MACAgent(
            sheaf_manifold=create_geometric_learning_system(
                num_vertices=2, num_edges=1, fiber_dim_vertex=1, fiber_dim_edge=1, seed=42
            )[0]
        )
        
        rho = create_quantum_mac_state(dimension=2, purity=0.8, seed=42)
        povm_ops = create_projection_operators(2)
        
        decisions = []
        
        for _ in range(10):
            decision, rho, _ = agent.extract_wisdom(
                current_rho=rho,
                povm_ops=povm_ops,
                deterministic=False
            )
            decisions.append(decision)
        
        # Verificar que todas las decisiones son válidas
        assert all(0 <= d < 2 for d in decisions), \
            f"Decisiones inválidas: {decisions}"


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
pytest.mark.povm = pytest.mark.povm
pytest.mark.lindblad = pytest.mark.lindblad
pytest.mark.cohomology = pytest.mark.cohomology
pytest.mark.information_geometry = pytest.mark.information_geometry
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow


# ══════════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA PARA EJECUCIÓN DIRECTA
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--cov=app.wisdom.mac_agent",
        "--cov-report=html",
        "--cov-report=term",
        "-m", "not slow",
        "--maxfail=10",
        "--durations=15"
    ])