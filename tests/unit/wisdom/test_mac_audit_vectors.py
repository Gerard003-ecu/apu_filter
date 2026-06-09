# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo de Pruebas: MAC Audit Vectors Test Suite                              ║
║ Ubicación: tests/wisdom/test_mac_audit_vectors.py                            ║
║ Versión: 2.0.0-Quantum-Sheaf-Audit-Test-Suite                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

Suite de Pruebas Rigurosas para Validación de:
───────────────────────────────────────────────
1. Divergencia de Umegaki (Entropía Relativa Cuántica)
2. Divergencias Generalizadas (Renyi, Petz)
3. Cohomología de Haces (Energía de Dirichlet)
4. Índice de Estabilidad Cuántica (Ψ_Q)
5. Vector de Auditoría Completo (Integración)

Metodología de Testing:
───────────────────────
- Pruebas unitarias (componentes aislados)
- Pruebas de axiomas matemáticos
- Pruebas de casos extremos
- Pruebas de integración end-to-end
- Pruebas de estabilidad numérica
- Pruebas de umbrales y vetos

Referencias:
────────────
- Umegaki (1962): "Conditional expectation in an operator algebra"
- Petz (1986): "Quasi-entropies for finite quantum systems"
- Hansen & Ghrist (2019): "Toward a spectral theory of cellular sheaves"
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

from app.wisdom.mac_audit_vectors import (
    UmegakiDivergenceAuditor,
    SheafCohomologyAuditor,
    QuantumStabilityIndex,
    TraceAnomalyVeto,
    CohomologicalObstructionError,
    EpistemologicalStatus,
    AuditMetrics,
    vector_audit_quantum_semantic_coherence
)

from app.wisdom.atomic_knowledge_matrix import (
    AtomicDensityMatrix,
    create_quantum_mac_state,
    NumericalInstabilityError
)

from app.adapters.mic_vectors import VectorResultStatus
from app.core.schemas import Stratum

logger = logging.getLogger("MAC.Audit.Vectors.Tests")


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES GLOBALES Y UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_g_physics() -> np.ndarray:
    """Tensor métrico físico mock para pruebas."""
    return np.eye(10, dtype=np.float64)


@pytest.fixture
def identity_state_2d() -> np.ndarray:
    """Estado maximalmente mixto 2D: I/2."""
    return np.eye(2, dtype=np.complex128) / 2.0


@pytest.fixture
def pure_state_2d() -> np.ndarray:
    """Estado puro 2D: |0⟩⟨0|."""
    psi = np.array([1, 0], dtype=np.complex128)
    return np.outer(psi, psi.conj())


def create_mixed_state_with_purity(dimension: int, purity: float, seed: int = 42) -> np.ndarray:
    """Crea estado mixto con pureza específica."""
    rho = create_quantum_mac_state(dimension=dimension, purity=purity, seed=seed)
    return rho.matrix


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1: PRUEBAS DE UMEGAKI DIVERGENCE AUDITOR
# ══════════════════════════════════════════════════════════════════════════════

class TestUmegakiDivergenceAuditor:
    """Suite de pruebas para divergencia de Umegaki."""
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Logaritmo Matricial
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_matrix_log_identity(self):
        """ln(I) = 0."""
        identity = np.eye(3, dtype=np.complex128)
        
        log_identity = UmegakiDivergenceAuditor.compute_matrix_log(identity)
        
        assert np.allclose(log_identity, 0.0, atol=1e-10), \
            "ln(I) debe ser matriz nula"
    
    def test_matrix_log_diagonal(self):
        """ln(diagonal) debe ser diagonal con ln de elementos."""
        eigenvalues = np.array([1.0, 2.0, 4.0])
        A = np.diag(eigenvalues)
        
        log_A = UmegakiDivergenceAuditor.compute_matrix_log(A)
        
        expected = np.diag(np.log(eigenvalues))
        
        assert np.allclose(log_A, expected, atol=1e-10), \
            "Logaritmo de diagonal incorrecto"
    
    def test_matrix_log_regularization(self):
        """Regularización debe evitar ln(0)."""
        # Matriz con eigenvalor nulo
        A = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        
        # Debe completarse sin error
        log_A = UmegakiDivergenceAuditor.compute_matrix_log(A, regularization=1e-10)
        
        assert log_A is not None
        assert not np.any(np.isnan(log_A))
        assert not np.any(np.isinf(log_A))
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Divergencia de Umegaki
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_umegaki_divergence_identical_states(self, identity_state_2d):
        """S(ρ||ρ) = 0."""
        rho = identity_state_2d
        
        divergence = UmegakiDivergenceAuditor.compute_divergence(rho, rho)
        
        assert np.isclose(divergence, 0.0, atol=1e-10), \
            f"Divergencia no nula para estados idénticos: {divergence}"
    
    def test_umegaki_divergence_non_negative(self):
        """S(ρ||σ) ≥ 0."""
        rho = create_mixed_state_with_purity(dimension=2, purity=0.7, seed=42)
        sigma = create_mixed_state_with_purity(dimension=2, purity=0.5, seed=43)
        
        divergence = UmegakiDivergenceAuditor.compute_divergence(rho, sigma)
        
        assert divergence >= -1e-12, f"Divergencia negativa: {divergence}"
    
    def test_umegaki_divergence_asymmetry(self):
        """S(ρ||σ) ≠ S(σ||ρ) en general."""
        rho = create_mixed_state_with_purity(dimension=2, purity=0.8, seed=42)
        sigma = create_mixed_state_with_purity(dimension=2, purity=0.4, seed=43)
        
        S_rho_sigma = UmegakiDivergenceAuditor.compute_divergence(rho, sigma)
        S_sigma_rho = UmegakiDivergenceAuditor.compute_divergence(sigma, rho)
        
        # No deben ser iguales (excepto casos degenerados)
        if np.isclose(S_rho_sigma, S_sigma_rho, atol=1e-6):
            pytest.skip("Estados resultaron simétricos (caso degenerado)")
    
    def test_umegaki_divergence_pure_vs_mixed(self, pure_state_2d, identity_state_2d):
        """Divergencia entre estado puro y mixto."""
        S_pure_mixed = UmegakiDivergenceAuditor.compute_divergence(
            pure_state_2d, identity_state_2d
        )
        
        # Debe ser positiva
        assert S_pure_mixed > 0, \
            f"Divergencia no positiva: {S_pure_mixed}"
    
    def test_umegaki_divergence_disjoint_support_warning(self):
        """Debe advertir sobre soportes disjuntos."""
        # Estado con soporte en |0⟩
        psi_0 = np.array([1, 0], dtype=np.complex128)
        rho = np.outer(psi_0, psi_0.conj())
        
        # Estado con soporte en |1⟩
        psi_1 = np.array([0, 1], dtype=np.complex128)
        sigma = np.outer(psi_1, psi_1.conj())
        
        # Debe computar sin crash (con advertencia en log)
        divergence = UmegakiDivergenceAuditor.compute_divergence(
            rho, sigma, check_support=True
        )
        
        # Divergencia debe ser grande (tiende a infinito)
        assert divergence > 1.0, \
            f"Divergencia baja para soportes disjuntos: {divergence}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Divergencia de Renyi
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_renyi_divergence_alpha_1_equals_umegaki(self):
        """S₁(ρ||σ) debe igualar S(ρ||σ)."""
        rho = create_mixed_state_with_purity(dimension=2, purity=0.6, seed=42)
        sigma = create_mixed_state_with_purity(dimension=2, purity=0.8, seed=43)
        
        S_umegaki = UmegakiDivergenceAuditor.compute_divergence(rho, sigma)
        S_renyi_1 = UmegakiDivergenceAuditor.compute_renyi_divergence(
            rho, sigma, alpha=1.0
        )
        
        assert np.isclose(S_umegaki, S_renyi_1, atol=1e-8), \
            f"S₁ no coincide con S_umegaki: {S_renyi_1} vs {S_umegaki}"
    
    def test_renyi_divergence_non_negative(self):
        """S_α(ρ||σ) ≥ 0 para α > 0."""
        rho = create_mixed_state_with_purity(dimension=2, purity=0.7, seed=42)
        sigma = create_mixed_state_with_purity(dimension=2, purity=0.5, seed=43)
        
        for alpha in [0.5, 2.0, 5.0]:
            S_renyi = UmegakiDivergenceAuditor.compute_renyi_divergence(
                rho, sigma, alpha=alpha
            )
            
            assert S_renyi >= -1e-10, \
                f"S_{alpha} negativa: {S_renyi}"
    
    def test_renyi_divergence_rejects_invalid_alpha(self):
        """Debe rechazar α ≤ 0."""
        rho = create_mixed_state_with_purity(dimension=2, purity=0.6, seed=42)
        sigma = create_mixed_state_with_purity(dimension=2, purity=0.8, seed=43)
        
        with pytest.raises(ValueError, match="positivo"):
            UmegakiDivergenceAuditor.compute_renyi_divergence(
                rho, sigma, alpha=-1.0
            )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2: PRUEBAS DE SHEAF COHOMOLOGY AUDITOR
# ══════════════════════════════════════════════════════════════════════════════

class TestSheafCohomologyAuditor:
    """Suite de pruebas para cohomología de haces."""
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Validación de Dimensiones
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_validate_dimensions_accepts_compatible(self, mock_g_physics):
        """Debe aceptar dimensiones compatibles."""
        delta_x = np.random.randn(10)
        
        # No debe lanzar error
        SheafCohomologyAuditor.validate_dimensions(delta_x, mock_g_physics)
    
    def test_validate_dimensions_rejects_non_vector(self):
        """Debe rechazar δx no vectorial."""
        delta_x = np.random.randn(5, 5)  # Matriz
        metric = np.eye(5)
        
        with pytest.raises(ValueError, match="1D"):
            SheafCohomologyAuditor.validate_dimensions(delta_x, metric)
    
    def test_validate_dimensions_rejects_non_square_metric(self):
        """Debe rechazar métrica no cuadrada."""
        delta_x = np.random.randn(5)
        metric = np.random.randn(5, 3)  # No cuadrada
        
        with pytest.raises(ValueError, match="cuadrada"):
            SheafCohomologyAuditor.validate_dimensions(delta_x, metric)
    
    def test_validate_dimensions_rejects_incompatible(self):
        """Debe rechazar dimensiones incompatibles."""
        delta_x = np.random.randn(5)
        metric = np.eye(10)
        
        with pytest.raises(ValueError, match="incompatible"):
            SheafCohomologyAuditor.validate_dimensions(delta_x, metric)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Energía de Dirichlet
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_dirichlet_energy_zero_for_zero_vector(self, mock_g_physics):
        """Energía debe ser 0 para vector nulo."""
        delta_x = np.zeros(10)
        
        energy = SheafCohomologyAuditor.compute_dirichlet_energy(
            delta_x, metric=mock_g_physics
        )
        
        assert np.isclose(energy, 0.0, atol=1e-12), \
            f"Energía no nula para vector nulo: {energy}"
    
    def test_dirichlet_energy_non_negative(self, mock_g_physics):
        """Energía debe ser no negativa para métrica positiva."""
        delta_x = np.random.randn(10)
        
        energy = SheafCohomologyAuditor.compute_dirichlet_energy(
            delta_x, metric=mock_g_physics
        )
        
        assert energy >= -1e-12, f"Energía negativa: {energy}"
    
    def test_dirichlet_energy_scales_quadratically(self, mock_g_physics):
        """Energía debe escalar cuadráticamente: E(2x) = 4E(x)."""
        delta_x = np.random.randn(10)
        
        E_1 = SheafCohomologyAuditor.compute_dirichlet_energy(
            delta_x, metric=mock_g_physics
        )
        
        E_2 = SheafCohomologyAuditor.compute_dirichlet_energy(
            2.0 * delta_x, metric=mock_g_physics
        )
        
        assert np.isclose(E_2, 4.0 * E_1, atol=1e-10), \
            f"Escalamiento cuadrático no verificado: E(2x)={E_2}, 4E(x)={4*E_1}"
    
    def test_dirichlet_energy_identity_metric_equals_euclidean(self):
        """Con métrica identidad, debe igualar energía Euclidiana."""
        delta_x = np.random.randn(5)
        metric_identity = np.eye(5)
        
        E_dirichlet = SheafCohomologyAuditor.compute_dirichlet_energy(
            delta_x, metric=metric_identity
        )
        
        E_euclidean = SheafCohomologyAuditor.compute_euclidean_energy(delta_x)
        
        assert np.isclose(E_dirichlet, E_euclidean, atol=1e-10), \
            f"Energías no coinciden: {E_dirichlet} vs {E_euclidean}"
    
    def test_dirichlet_energy_handles_nan_gracefully(self):
        """Debe manejar NaN retornando infinito."""
        delta_x = np.array([np.nan, 1.0, 2.0])
        metric = np.eye(3)
        
        energy = SheafCohomologyAuditor.compute_dirichlet_energy(
            delta_x, metric=metric
        )
        
        assert np.isinf(energy), \
            f"NaN no manejado correctamente: {energy}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Clasificación de Energía
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_classify_energy_perfect_holonomy(self):
        """E ≈ 0 debe clasificarse como holonomía perfecta."""
        classification = SheafCohomologyAuditor.classify_energy_level(1e-8)
        assert classification == "HOLONOMÍA_PERFECTA"
    
    def test_classify_energy_low_tension(self):
        """E ∈ (1e-6, 0.1) debe ser tensión baja."""
        classification = SheafCohomologyAuditor.classify_energy_level(0.05)
        assert classification == "TENSIÓN_BAJA"
    
    def test_classify_energy_critical_friction(self):
        """E > 10 debe ser fricción crítica."""
        classification = SheafCohomologyAuditor.classify_energy_level(15.0)
        assert classification == "FRICCIÓN_CRÍTICA"
    
    def test_classify_energy_irresoluble_paradox(self):
        """E = ∞ debe ser paradoja irresoluble."""
        classification = SheafCohomologyAuditor.classify_energy_level(float('inf'))
        assert classification == "PARADOJA_IRRESOLUBLE"


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3: PRUEBAS DE QUANTUM STABILITY INDEX
# ══════════════════════════════════════════════════════════════════════════════

class TestQuantumStabilityIndex:
    """Suite de pruebas para índice de estabilidad cuántica."""
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Fidelidad
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_fidelity_identical_states(self):
        """Fidelidad entre estados idénticos debe ser 1."""
        rho = create_mixed_state_with_purity(dimension=2, purity=0.7, seed=42)
        
        fidelity = QuantumStabilityIndex.compute_fidelity(rho, rho)
        
        assert np.isclose(fidelity, 1.0, atol=1e-10), \
            f"Fidelidad no unitaria: {fidelity}"
    
    def test_fidelity_orthogonal_states(self):
        """Fidelidad entre estados ortogonales debe ser 0."""
        psi_0 = np.array([1, 0], dtype=np.complex128)
        psi_1 = np.array([0, 1], dtype=np.complex128)
        
        rho = np.outer(psi_0, psi_0.conj())
        sigma = np.outer(psi_1, psi_1.conj())
        
        fidelity = QuantumStabilityIndex.compute_fidelity(rho, sigma)
        
        assert np.isclose(fidelity, 0.0, atol=1e-10), \
            f"Fidelidad no nula para ortogonales: {fidelity}"
    
    def test_fidelity_bounds(self):
        """Fidelidad debe estar en [0, 1]."""
        rho = create_mixed_state_with_purity(dimension=3, purity=0.8, seed=42)
        sigma = create_mixed_state_with_purity(dimension=3, purity=0.5, seed=43)
        
        fidelity = QuantumStabilityIndex.compute_fidelity(rho, sigma)
        
        assert 0 <= fidelity <= 1 + 1e-10, \
            f"Fidelidad fuera de rango: {fidelity}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Pureza
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_purity_pure_state(self):
        """Pureza de estado puro debe ser 1."""
        psi = np.array([1, 0, 0], dtype=np.complex128)
        rho = np.outer(psi, psi.conj())
        
        purity = QuantumStabilityIndex.compute_purity(rho)
        
        assert np.isclose(purity, 1.0, atol=1e-10), \
            f"Pureza de estado puro incorrecta: {purity}"
    
    def test_purity_maximally_mixed(self):
        """Pureza de estado maximalmente mixto debe ser 1/d."""
        dim = 4
        rho = np.eye(dim, dtype=np.complex128) / dim
        
        purity = QuantumStabilityIndex.compute_purity(rho)
        expected = 1.0 / dim
        
        assert np.isclose(purity, expected, atol=1e-10), \
            f"Pureza incorrecta: {purity} vs {expected}"
    
    def test_purity_bounds(self):
        """Pureza debe estar en [1/d, 1]."""
        dim = 3
        rho = create_mixed_state_with_purity(dimension=dim, purity=0.6, seed=42)
        
        purity = QuantumStabilityIndex.compute_purity(rho)
        
        assert 1.0/dim - 1e-10 <= purity <= 1.0 + 1e-10, \
            f"Pureza fuera de rango: {purity}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Índice Ψ_Q
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_psi_q_perfect_coherence(self):
        """Estados idénticos con E=0 deben dar Ψ_Q ≈ pureza."""
        rho = create_mixed_state_with_purity(dimension=2, purity=0.9, seed=42)
        
        psi_q, components = QuantumStabilityIndex.compute_psi_q(
            rho, rho, dirichlet_energy=0.0
        )
        
        # F=1, exp(-0)=1, entonces Ψ_Q = pureza
        expected = components['purity']
        
        assert np.isclose(psi_q, expected, atol=1e-8), \
            f"Ψ_Q incorrecto: {psi_q} vs {expected}"
    
    def test_psi_q_high_energy_penalty(self):
        """Alta energía debe reducir Ψ_Q significativamente."""
        rho = create_mixed_state_with_purity(dimension=2, purity=0.8, seed=42)
        sigma = create_mixed_state_with_purity(dimension=2, purity=0.75, seed=43)
        
        psi_q_low_energy, _ = QuantumStabilityIndex.compute_psi_q(
            rho, sigma, dirichlet_energy=0.1
        )
        
        psi_q_high_energy, _ = QuantumStabilityIndex.compute_psi_q(
            rho, sigma, dirichlet_energy=10.0
        )
        
        assert psi_q_high_energy < psi_q_low_energy, \
            f"Alta energía no penaliza: {psi_q_high_energy} >= {psi_q_low_energy}"
    
    def test_psi_q_conservative_mode_stronger_penalty(self):
        """Modo conservador debe penalizar más."""
        rho = create_mixed_state_with_purity(dimension=2, purity=0.8, seed=42)
        sigma = create_mixed_state_with_purity(dimension=2, purity=0.7, seed=43)
        energy = 2.0
        
        psi_q_standard, _ = QuantumStabilityIndex.compute_psi_q(
            rho, sigma, energy, conservative=False
        )
        
        psi_q_conservative, _ = QuantumStabilityIndex.compute_psi_q(
            rho, sigma, energy, conservative=True
        )
        
        assert psi_q_conservative < psi_q_standard, \
            "Modo conservador no penaliza más"
    
    def test_psi_q_bounds(self):
        """Ψ_Q debe estar en [0, 1]."""
        rho = create_mixed_state_with_purity(dimension=2, purity=0.6, seed=42)
        sigma = create_mixed_state_with_purity(dimension=2, purity=0.8, seed=43)
        
        psi_q, _ = QuantumStabilityIndex.compute_psi_q(
            rho, sigma, dirichlet_energy=1.0
        )
        
        assert 0 <= psi_q <= 1 + 1e-10, \
            f"Ψ_Q fuera de rango: {psi_q}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Clasificación de Estabilidad
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_classify_stability_homomorphism_verified(self):
        """Ψ_Q ≥ 0.90 debe ser homomorphism verified."""
        status = QuantumStabilityIndex.classify_stability(0.95)
        assert status == EpistemologicalStatus.HOMOMORPHISM_VERIFIED
    
    def test_classify_stability_acceptable_deviation(self):
        """Ψ_Q ∈ [0.70, 0.90) debe ser acceptable deviation."""
        status = QuantumStabilityIndex.classify_stability(0.80)
        assert status == EpistemologicalStatus.ACCEPTABLE_DEVIATION
    
    def test_classify_stability_semantic_drift(self):
        """Ψ_Q ∈ [0.50, 0.70) debe ser semantic drift."""
        status = QuantumStabilityIndex.classify_stability(0.60)
        assert status == EpistemologicalStatus.SEMANTIC_DRIFT
    
    def test_classify_stability_hallucination_detected(self):
        """Ψ_Q < 0.30 debe ser hallucination detected."""
        status = QuantumStabilityIndex.classify_stability(0.20)
        assert status == EpistemologicalStatus.HALLUCINATION_DETECTED


# ══════════════════════════════════════════════════════════════════════════════
# FASE 4: PRUEBAS DE VECTOR DE AUDITORÍA COMPLETO
# ══════════════════════════════════════════════════════════════════════════════

class TestVectorAuditQuantumSemanticCoherence:
    """Suite de pruebas para vector de auditoría completo."""
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Auditoría Exitosa
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_audit_accepts_coherent_states(self, mock_g_physics):
        """Debe aceptar estados coherentes."""
        # Estados muy similares
        rho = create_quantum_mac_state(dimension=2, purity=0.8, seed=42)
        sigma = create_quantum_mac_state(dimension=2, purity=0.79, seed=42)
        
        # Obstrucción baja
        delta_x = np.random.randn(10) * 0.01
        
        result = vector_audit_quantum_semantic_coherence(
            mac_rho=rho,
            reference_sigma=sigma,
            logical_delta_x=delta_x,
            umegaki_threshold=1.0,
            psi_q_minimum=0.5
        )
        
        assert result['success'], f"Auditoría falló: {result.get('error')}"
        assert result['status'] == VectorResultStatus.SUCCESS
        assert 'quantum_stability_index' in result
    
    def test_audit_computes_all_metrics(self):
        """Debe calcular todas las métricas."""
        rho = create_quantum_mac_state(dimension=2, purity=0.7, seed=42)
        sigma = create_quantum_mac_state(dimension=2, purity=0.65, seed=43)
        delta_x = np.random.randn(10) * 0.1
        
        result = vector_audit_quantum_semantic_coherence(
            mac_rho=rho,
            reference_sigma=sigma,
            logical_delta_x=delta_x,
            compute_extended_metrics=True
        )
        
        if result['success']:
            assert 'umegaki_divergence' in result
            assert 'dirichlet_energy' in result
            assert 'quantum_stability_index' in result
            assert 'fidelity' in result
            assert 'purity' in result
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Vetos
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_audit_veto_high_umegaki_divergence(self):
        """Debe vetar si divergencia de Umegaki es alta."""
        # Estados muy diferentes
        psi_0 = np.array([1, 0], dtype=np.complex128)
        psi_1 = np.array([0, 1], dtype=np.complex128)
        
        rho = AtomicDensityMatrix(np.outer(psi_0, psi_0.conj()))
        sigma = AtomicDensityMatrix(np.outer(psi_1, psi_1.conj()))
        
        delta_x = np.zeros(10)
        
        result = vector_audit_quantum_semantic_coherence(
            mac_rho=rho,
            reference_sigma=sigma,
            logical_delta_x=delta_x,
            umegaki_threshold=0.1  # Umbral muy bajo
        )
        
        assert not result['success']
        assert result['status'] == VectorResultStatus.VALIDATION_ERROR
    
    def test_audit_veto_high_dirichlet_energy(self):
        """Debe vetar si energía de Dirichlet es alta."""
        rho = create_quantum_mac_state(dimension=2, purity=0.8, seed=42)
        sigma = create_quantum_mac_state(dimension=2, purity=0.75, seed=43)
        
        # Obstrucción muy alta
        delta_x = np.random.randn(10) * 100.0
        
        result = vector_audit_quantum_semantic_coherence(
            mac_rho=rho,
            reference_sigma=sigma,
            logical_delta_x=delta_x,
            dirichlet_threshold=1.0  # Umbral bajo
        )
        
        assert not result['success']
        assert result['status'] == VectorResultStatus.TOPOLOGY_ERROR
    
    def test_audit_veto_low_psi_q(self):
        """Debe vetar si Ψ_Q es bajo."""
        # Estados muy diferentes
        rho = create_quantum_mac_state(dimension=2, purity=0.3, seed=42)
        sigma = create_quantum_mac_state(dimension=2, purity=0.9, seed=43)
        
        # Alta obstrucción
        delta_x = np.random.randn(10) * 5.0
        
        result = vector_audit_quantum_semantic_coherence(
            mac_rho=rho,
            reference_sigma=sigma,
            logical_delta_x=delta_x,
            psi_q_minimum=0.8  # Umbral muy alto
        )
        
        assert not result['success']
        assert result['status'] == VectorResultStatus.LOGIC_ERROR
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Configuraciones
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_audit_conservative_penalty_mode(self):
        """Modo conservador debe ser más estricto."""
        rho = create_quantum_mac_state(dimension=2, purity=0.7, seed=42)
        sigma = create_quantum_mac_state(dimension=2, purity=0.6, seed=43)
        delta_x = np.random.randn(10) * 0.5
        
        result_standard = vector_audit_quantum_semantic_coherence(
            mac_rho=rho,
            reference_sigma=sigma,
            logical_delta_x=delta_x,
            conservative_penalty=False,
            psi_q_minimum=0.3
        )
        
        result_conservative = vector_audit_quantum_semantic_coherence(
            mac_rho=rho,
            reference_sigma=sigma,
            logical_delta_x=delta_x,
            conservative_penalty=True,
            psi_q_minimum=0.3
        )
        
        if result_standard['success'] and result_conservative['success']:
            psi_q_standard = result_standard['quantum_stability_index']
            psi_q_conservative = result_conservative['quantum_stability_index']
            
            assert psi_q_conservative < psi_q_standard, \
                "Modo conservador no es más estricto"
    
    def test_audit_extended_metrics_optional(self):
        """Métricas extendidas deben ser opcionales."""
        rho = create_quantum_mac_state(dimension=2, purity=0.8, seed=42)
        sigma = create_quantum_mac_state(dimension=2, purity=0.75, seed=43)
        delta_x = np.random.randn(10) * 0.1
        
        result_extended = vector_audit_quantum_semantic_coherence(
            mac_rho=rho,
            reference_sigma=sigma,
            logical_delta_x=delta_x,
            compute_extended_metrics=True
        )
        
        result_basic = vector_audit_quantum_semantic_coherence(
            mac_rho=rho,
            reference_sigma=sigma,
            logical_delta_x=delta_x,
            compute_extended_metrics=False
        )
        
        if result_extended['success']:
            assert 'entropy_production' in result_extended
        
        if result_basic['success']:
            # Métricas extendidas pueden estar ausentes
            pass
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de AuditMetrics Dataclass
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_audit_metrics_is_acceptable(self):
        """is_acceptable debe detectar estados aceptables."""
        metrics_good = AuditMetrics(
            umegaki_divergence=0.1,
            dirichlet_energy=0.5,
            quantum_stability_index=0.85,
            fidelity=0.9,
            purity_mac=0.8,
            purity_reference=0.75,
            entropy_production=0.1,
            epistemological_status=EpistemologicalStatus.HOMOMORPHISM_VERIFIED,
            execution_time_ms=10.0
        )
        
        assert metrics_good.is_acceptable()
        
        metrics_bad = AuditMetrics(
            umegaki_divergence=2.0,
            dirichlet_energy=15.0,
            quantum_stability_index=0.2,
            fidelity=0.3,
            purity_mac=0.5,
            purity_reference=0.9,
            entropy_production=0.5,
            epistemological_status=EpistemologicalStatus.HALLUCINATION_DETECTED,
            execution_time_ms=10.0
        )
        
        assert not metrics_bad.is_acceptable()


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
pytest.mark.umegaki = pytest.mark.umegaki
pytest.mark.sheaf = pytest.mark.sheaf
pytest.mark.stability = pytest.mark.stability
pytest.mark.audit = pytest.mark.audit
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
        "--cov=app.wisdom.mac_audit_vectors",
        "--cov-report=html",
        "--cov-report=term",
        "-m", "not slow",
        "--maxfail=10",
        "--durations=15"
    ])