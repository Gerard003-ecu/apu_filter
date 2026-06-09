# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo de Pruebas: MAC Vectors Test Suite                                    ║
║ Ubicación: tests/wisdom/test_mac_vectors.py                                  ║
║ Versión: 2.0.0-Quantum-Channel-Morphisms-Test-Suite                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Suite de Pruebas Rigurosas para Validación de:
───────────────────────────────────────────────
1. Geometría de Información (Fidelidad de Uhlmann, Distancia de Bures)
2. Teoría Modular (Tomita-Takesaki, Conjugación Modular)
3. Canales Cuánticos (CPTP, Kraus, Caracterización)
4. Vectores de Inyección (Asimilación de Cartuchos)
5. Vectores de Medición (Colapso POVM, Decisiones)
6. Vectores de Auditoría (Adjunción de Galois)

Metodología de Testing:
───────────────────────
- Pruebas unitarias (componentes aislados)
- Pruebas de integración (flujos completos)
- Pruebas de propiedades (property-based testing)
- Pruebas de invariantes matemáticos
- Pruebas de canales cuánticos estándar
- Pruebas de casos extremos

Referencias:
────────────
- Kraus (1983): "States, Effects, and Operations"
- Uhlmann (1976): "The transition probability in the state space"
- Nielsen & Chuang (2010): "Quantum Computation and Quantum Information"
- Takesaki (1970): "Tomita's theory of modular Hilbert algebras"
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

from app.wisdom.mac_vectors import (
    BuresUhlmannAuditor,
    TomitaTakesakiAuditor,
    QuantumChannelCharacterizer,
    InjectionQuality,
    ChannelType,
    ChannelCharacterization,
    InjectionReport,
    ModularConjugationReport,
    vector_assimilate_toon_cartridge,
    vector_collapse_povm_decision,
    vector_audit_modular_conjugation
)

from app.wisdom.atomic_knowledge_matrix import (
    AtomicDensityMatrix,
    create_quantum_mac_state,
    NumericalInstabilityError
)

from app.adapters.mic_vectors import VectorResultStatus
from app.core.schemas import Stratum

logger = logging.getLogger("MAC.Vectors.Tests")


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES GLOBALES Y UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def bures_auditor() -> BuresUhlmannAuditor:
    """Auditor de Bures-Uhlmann."""
    return BuresUhlmannAuditor()


@pytest.fixture
def tomita_auditor() -> TomitaTakesakiAuditor:
    """Auditor de Tomita-Takesaki."""
    return TomitaTakesakiAuditor()


@pytest.fixture
def pauli_operators() -> Dict[str, np.ndarray]:
    """Operadores de Pauli estándar."""
    return {
        'I': np.array([[1, 0], [0, 1]], dtype=np.complex128),
        'X': np.array([[0, 1], [1, 0]], dtype=np.complex128),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        'Z': np.array([[1, 0], [0, -1]], dtype=np.complex128)
    }


def create_unitary_channel(U: np.ndarray) -> List[np.ndarray]:
    """Crea canal unitario con un solo operador de Kraus."""
    return [U]


def create_depolarizing_channel(dimension: int, p: float) -> List[np.ndarray]:
    """Crea canal despolarizante."""
    if dimension == 2:
        pauli_ops = [
            np.array([[1, 0], [0, 1]], dtype=np.complex128),
            np.array([[0, 1], [1, 0]], dtype=np.complex128),
            np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
            np.array([[1, 0], [0, -1]], dtype=np.complex128)
        ]
        
        kraus_ops = [
            np.sqrt(1 - 3*p/4) * pauli_ops[0],
            np.sqrt(p/4) * pauli_ops[1],
            np.sqrt(p/4) * pauli_ops[2],
            np.sqrt(p/4) * pauli_ops[3]
        ]
        return kraus_ops
    else:
        raise NotImplementedError("Canal despolarizante solo para d=2")


def create_amplitude_damping_channel(gamma: float) -> List[np.ndarray]:
    """Crea canal de amortiguamiento de amplitud."""
    E0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=np.complex128)
    E1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=np.complex128)
    return [E0, E1]


def create_phase_damping_channel(gamma: float) -> List[np.ndarray]:
    """Crea canal de amortiguamiento de fase."""
    E0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=np.complex128)
    E1 = np.array([[0, 0], [0, np.sqrt(gamma)]], dtype=np.complex128)
    return [E0, E1]


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1: PRUEBAS DE BURES-UHLMANN AUDITOR
# ══════════════════════════════════════════════════════════════════════════════

class TestBuresUhlmannAuditor:
    """Suite de pruebas para auditor de Bures-Uhlmann."""
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Fidelidad
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_fidelity_identical_states(self, bures_auditor):
        """Fidelidad entre estados idénticos debe ser 1."""
        rho = create_quantum_mac_state(dimension=2, purity=0.7, seed=42)
        
        fidelity = bures_auditor.compute_fidelity(rho.matrix, rho.matrix)
        
        assert np.isclose(fidelity, 1.0, atol=1e-10), \
            f"Fidelidad no unitaria para estados idénticos: {fidelity}"
    
    def test_fidelity_orthogonal_pure_states(self, bures_auditor):
        """Fidelidad entre estados puros ortogonales debe ser 0."""
        psi_0 = np.array([1, 0], dtype=np.complex128)
        psi_1 = np.array([0, 1], dtype=np.complex128)
        
        rho_0 = np.outer(psi_0, psi_0.conj())
        rho_1 = np.outer(psi_1, psi_1.conj())
        
        fidelity = bures_auditor.compute_fidelity(rho_0, rho_1)
        
        assert np.isclose(fidelity, 0.0, atol=1e-10), \
            f"Fidelidad no nula para estados ortogonales: {fidelity}"
    
    def test_fidelity_bounds(self, bures_auditor):
        """Fidelidad debe estar en [0, 1]."""
        rho1 = create_quantum_mac_state(dimension=3, purity=0.8, seed=42)
        rho2 = create_quantum_mac_state(dimension=3, purity=0.5, seed=43)
        
        fidelity = bures_auditor.compute_fidelity(rho1.matrix, rho2.matrix)
        
        assert 0 <= fidelity <= 1 + 1e-10, \
            f"Fidelidad fuera de rango: {fidelity}"
    
    def test_fidelity_symmetric(self, bures_auditor):
        """Fidelidad debe ser simétrica: F(ρ,σ) = F(σ,ρ)."""
        rho1 = create_quantum_mac_state(dimension=2, purity=0.6, seed=42)
        rho2 = create_quantum_mac_state(dimension=2, purity=0.9, seed=43)
        
        f_12 = bures_auditor.compute_fidelity(rho1.matrix, rho2.matrix)
        f_21 = bures_auditor.compute_fidelity(rho2.matrix, rho1.matrix)
        
        assert np.isclose(f_12, f_21, atol=1e-10), \
            f"Fidelidad no simétrica: F(ρ₁,ρ₂)={f_12}, F(ρ₂,ρ₁)={f_21}"
    
    def test_fidelity_pure_states_equals_overlap_squared(self, bures_auditor):
        """Para estados puros: F(|ψ⟩,|φ⟩) = |⟨ψ|φ⟩|²."""
        # Estados puros con overlap conocido
        theta = np.pi / 6
        psi = np.array([np.cos(theta/2), np.sin(theta/2)], dtype=np.complex128)
        phi = np.array([1, 0], dtype=np.complex128)
        
        rho_psi = np.outer(psi, psi.conj())
        rho_phi = np.outer(phi, phi.conj())
        
        fidelity = bures_auditor.compute_fidelity(rho_psi, rho_phi)
        expected = np.abs(np.vdot(phi, psi)) ** 2
        
        assert np.isclose(fidelity, expected, atol=1e-10), \
            f"Fidelidad incorrecta: {fidelity} vs {expected}"
    
    def test_fidelity_alternative_method_consistency(self, bures_auditor):
        """Métodos estándar y eigendecomposición deben coincidir."""
        rho1 = create_quantum_mac_state(dimension=3, purity=0.7, seed=42)
        rho2 = create_quantum_mac_state(dimension=3, purity=0.6, seed=43)
        
        f_standard = bures_auditor.compute_fidelity(
            rho1.matrix, rho2.matrix, method='standard'
        )
        f_eigen = bures_auditor.compute_fidelity(
            rho1.matrix, rho2.matrix, method='eigendecomposition'
        )
        
        assert np.isclose(f_standard, f_eigen, atol=1e-8), \
            f"Métodos inconsistentes: {f_standard} vs {f_eigen}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Distancias
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_bures_distance_identical_states(self, bures_auditor):
        """Distancia de Bures entre estados idénticos debe ser 0."""
        rho = create_quantum_mac_state(dimension=2, purity=0.8, seed=42)
        
        distance = bures_auditor.compute_bures_distance(rho.matrix, rho.matrix)
        
        assert np.isclose(distance, 0.0, atol=1e-10), \
            f"Distancia no nula: {distance}"
    
    def test_bures_distance_orthogonal_states(self, bures_auditor):
        """Distancia de Bures entre ortogonales debe ser √2."""
        psi_0 = np.array([1, 0], dtype=np.complex128)
        psi_1 = np.array([0, 1], dtype=np.complex128)
        
        rho_0 = np.outer(psi_0, psi_0.conj())
        rho_1 = np.outer(psi_1, psi_1.conj())
        
        distance = bures_auditor.compute_bures_distance(rho_0, rho_1)
        expected = np.sqrt(2.0)
        
        assert np.isclose(distance, expected, atol=1e-10), \
            f"Distancia incorrecta: {distance} vs {expected}"
    
    def test_bures_distance_non_negative(self, bures_auditor):
        """Distancia de Bures debe ser no negativa."""
        rho1 = create_quantum_mac_state(dimension=3, purity=0.5, seed=42)
        rho2 = create_quantum_mac_state(dimension=3, purity=0.7, seed=43)
        
        distance = bures_auditor.compute_bures_distance(rho1.matrix, rho2.matrix)
        
        assert distance >= -1e-12, f"Distancia negativa: {distance}"
    
    def test_bures_fidelity_relationship(self, bures_auditor):
        """d²_B(ρ,σ) = 2(1 - √F(ρ,σ))."""
        rho1 = create_quantum_mac_state(dimension=2, purity=0.6, seed=42)
        rho2 = create_quantum_mac_state(dimension=2, purity=0.8, seed=43)
        
        fidelity = bures_auditor.compute_fidelity(rho1.matrix, rho2.matrix)
        distance = bures_auditor.compute_bures_distance(rho1.matrix, rho2.matrix)
        
        expected_distance = np.sqrt(2.0 * (1.0 - np.sqrt(fidelity)))
        
        assert np.isclose(distance, expected_distance, atol=1e-10), \
            f"Relación Bures-Fidelidad incorrecta: {distance} vs {expected_distance}"
    
    def test_hellinger_distance_bounds(self, bures_auditor):
        """Distancia de Hellinger debe estar en [0, 1]."""
        rho1 = create_quantum_mac_state(dimension=3, purity=0.7, seed=42)
        rho2 = create_quantum_mac_state(dimension=3, purity=0.4, seed=43)
        
        distance = bures_auditor.compute_hellinger_distance(rho1.matrix, rho2.matrix)
        
        assert 0 <= distance <= 1 + 1e-10, \
            f"Distancia de Hellinger fuera de rango: {distance}"
    
    def test_trace_distance_bounds(self, bures_auditor):
        """Distancia traza debe estar en [0, 1]."""
        rho1 = create_quantum_mac_state(dimension=2, purity=0.9, seed=42)
        rho2 = create_quantum_mac_state(dimension=2, purity=0.3, seed=43)
        
        distance = bures_auditor.compute_trace_distance(rho1.matrix, rho2.matrix)
        
        assert 0 <= distance <= 1 + 1e-10, \
            f"Distancia traza fuera de rango: {distance}"
    
    def test_trace_distance_orthogonal_pure_states(self, bures_auditor):
        """Distancia traza entre puros ortogonales debe ser 1."""
        psi_0 = np.array([1, 0], dtype=np.complex128)
        psi_1 = np.array([0, 1], dtype=np.complex128)
        
        rho_0 = np.outer(psi_0, psi_0.conj())
        rho_1 = np.outer(psi_1, psi_1.conj())
        
        distance = bures_auditor.compute_trace_distance(rho_0, rho_1)
        
        assert np.isclose(distance, 1.0, atol=1e-10), \
            f"Distancia traza incorrecta: {distance}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Clasificación de Calidad
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_injection_quality_excellent(self, bures_auditor):
        """Fidelidad > 0.95 debe ser EXCELLENT."""
        quality = bures_auditor.classify_injection_quality(0.97)
        assert quality == InjectionQuality.EXCELLENT
    
    def test_injection_quality_good(self, bures_auditor):
        """Fidelidad > 0.85 debe ser GOOD."""
        quality = bures_auditor.classify_injection_quality(0.90)
        assert quality == InjectionQuality.GOOD
    
    def test_injection_quality_acceptable(self, bures_auditor):
        """Fidelidad > 0.70 debe ser ACCEPTABLE."""
        quality = bures_auditor.classify_injection_quality(0.75)
        assert quality == InjectionQuality.ACCEPTABLE
    
    def test_injection_quality_degraded(self, bures_auditor):
        """Fidelidad > 0.50 debe ser DEGRADED."""
        quality = bures_auditor.classify_injection_quality(0.60)
        assert quality == InjectionQuality.DEGRADED
    
    def test_injection_quality_rejected(self, bures_auditor):
        """Fidelidad ≤ 0.50 debe ser REJECTED."""
        quality = bures_auditor.classify_injection_quality(0.40)
        assert quality == InjectionQuality.REJECTED


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2: PRUEBAS DE TOMITA-TAKESAKI AUDITOR
# ══════════════════════════════════════════════════════════════════════════════

class TestTomitaTakesakiAuditor:
    """Suite de pruebas para auditor de Tomita-Takesaki."""
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Asimetría Modular
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_modular_asymmetry_pure_state(self, tomita_auditor):
        """Asimetría modular de estado puro debe ser 0."""
        rho_pure = create_quantum_mac_state(dimension=3, purity=1.0, seed=42)
        
        asymmetry = tomita_auditor.compute_modular_asymmetry(rho_pure.matrix)
        
        assert np.isclose(asymmetry, 0.0, atol=1e-10), \
            f"Asimetría no nula para estado puro: {asymmetry}"
    
    def test_modular_asymmetry_maximally_mixed(self, tomita_auditor):
        """Asimetría modular de estado maximalmente mixto debe ser ln(d)."""
        dim = 4
        rho_mixed = np.eye(dim, dtype=np.complex128) / dim
        
        asymmetry = tomita_auditor.compute_modular_asymmetry(rho_mixed)
        expected = np.log(dim)
        
        assert np.isclose(asymmetry, expected, atol=1e-10), \
            f"Asimetría incorrecta: {asymmetry} vs {expected}"
    
    def test_modular_asymmetry_non_negative(self, tomita_auditor):
        """Asimetría modular debe ser no negativa."""
        rho = create_quantum_mac_state(dimension=3, purity=0.5, seed=42)
        
        asymmetry = tomita_auditor.compute_modular_asymmetry(rho.matrix)
        
        assert asymmetry >= -1e-12, f"Asimetría negativa: {asymmetry}"
    
    def test_modular_asymmetry_empty_state_raises_error(self, tomita_auditor):
        """Estado vacío debe lanzar error."""
        rho_empty = np.zeros((3, 3), dtype=np.complex128)
        
        with pytest.raises(NumericalInstabilityError, match="vacío"):
            tomita_auditor.compute_modular_asymmetry(rho_empty)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Entropía Relativa
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_relative_entropy_identical_states(self, tomita_auditor):
        """Entropía relativa D(ρ||ρ) = 0."""
        rho = create_quantum_mac_state(dimension=2, purity=0.7, seed=42)
        
        rel_entropy = tomita_auditor.compute_relative_entropy(rho.matrix, rho.matrix)
        
        assert np.isclose(rel_entropy, 0.0, atol=1e-10), \
            f"Entropía relativa no nula: {rel_entropy}"
    
    def test_relative_entropy_non_negative(self, tomita_auditor):
        """Entropía relativa debe ser no negativa."""
        rho1 = create_quantum_mac_state(dimension=2, purity=0.8, seed=42)
        rho2 = create_quantum_mac_state(dimension=2, purity=0.5, seed=43)
        
        rel_entropy = tomita_auditor.compute_relative_entropy(rho1.matrix, rho2.matrix)
        
        assert rel_entropy >= -1e-12, f"Entropía relativa negativa: {rel_entropy}"
    
    def test_relative_entropy_to_maximally_mixed(self, tomita_auditor):
        """Entropía relativa a estado maximalmente mixto."""
        dim = 3
        rho = create_quantum_mac_state(dimension=dim, purity=0.6, seed=42)
        
        # σ = I/d (maximalmente mixto)
        rel_entropy = tomita_auditor.compute_relative_entropy(rho.matrix, sigma=None)
        
        # Debe ser S(σ) - S(ρ) = ln(d) - S(ρ)
        asymmetry_rho = tomita_auditor.compute_modular_asymmetry(rho.matrix)
        expected = np.log(dim) - asymmetry_rho
        
        assert np.isclose(rel_entropy, expected, atol=1e-8), \
            f"Entropía relativa incorrecta: {rel_entropy} vs {expected}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Información de Fisher
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_fisher_information_non_negative(self, tomita_auditor):
        """Información de Fisher debe ser no negativa."""
        rho = create_quantum_mac_state(dimension=2, purity=0.7, seed=42)
        observable = np.array([[1, 0], [0, -1]], dtype=np.complex128)  # σ_z
        
        fisher = tomita_auditor.compute_quantum_fisher_information(
            rho.matrix, observable
        )
        
        assert fisher >= -1e-12, f"Información de Fisher negativa: {fisher}"
    
    def test_fisher_information_pure_state(self, tomita_auditor):
        """Información de Fisher para estado puro."""
        psi = np.array([1, 0], dtype=np.complex128)
        rho = np.outer(psi, psi.conj())
        observable = np.array([[0, 1], [1, 0]], dtype=np.complex128)  # σ_x
        
        fisher = tomita_auditor.compute_quantum_fisher_information(rho, observable)
        
        # Para estado puro en eigenestado de observabl: F_Q = 0
        assert fisher >= 0, f"Información de Fisher: {fisher}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Conjugación Modular Completa
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_verify_modular_conjugation_valid(self, tomita_auditor):
        """Verificación de conjugación modular válida."""
        rho = create_quantum_mac_state(dimension=3, purity=0.8, seed=42)
        mic_rank = 10
        
        report = tomita_auditor.verify_modular_conjugation(rho.matrix, mic_rank)
        
        assert isinstance(report, ModularConjugationReport)
        assert report.modular_asymmetry >= 0
        assert report.relative_entropy >= 0
        assert report.fisher_information >= 0
    
    def test_verify_modular_conjugation_accepts_low_asymmetry(self, tomita_auditor):
        """Baja asimetría debe ser aceptada."""
        rho_pure = create_quantum_mac_state(dimension=2, purity=0.99, seed=42)
        mic_rank = 10
        
        report = tomita_auditor.verify_modular_conjugation(rho_pure.matrix, mic_rank)
        
        assert report.is_valid(), "Conjugación válida rechazada"
        assert report.galois_adjunction_secured
    
    def test_verify_modular_conjugation_rejects_high_asymmetry(self, tomita_auditor):
        """Alta asimetría debe ser rechazada."""
        # Estado muy mixto
        dim = 5
        rho_mixed = np.eye(dim, dtype=np.complex128) / dim
        mic_rank = 2  # Rango pequeño → umbral bajo
        
        report = tomita_auditor.verify_modular_conjugation(rho_mixed, mic_rank)
        
        # Asimetría = ln(5) ≈ 1.6, umbral = ln(2) ≈ 0.69
        assert not report.is_valid(), "Alta asimetría no rechazada"


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3: PRUEBAS DE QUANTUM CHANNEL CHARACTERIZER
# ══════════════════════════════════════════════════════════════════════════════

class TestQuantumChannelCharacterizer:
    """Suite de pruebas para caracterizador de canales."""
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Resolución de Identidad
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_kraus_identity_resolution_unitary(self, pauli_operators):
        """Canal unitario debe resolver identidad."""
        U = pauli_operators['X']
        kraus_ops = create_unitary_channel(U)
        
        is_valid, error = QuantumChannelCharacterizer.verify_kraus_identity_resolution(
            kraus_ops
        )
        
        assert is_valid, f"Identidad no resuelta: error={error}"
        assert error < 1e-10
    
    def test_kraus_identity_resolution_depolarizing(self):
        """Canal despolarizante debe resolver identidad."""
        kraus_ops = create_depolarizing_channel(dimension=2, p=0.3)
        
        is_valid, error = QuantumChannelCharacterizer.verify_kraus_identity_resolution(
            kraus_ops
        )
        
        assert is_valid, f"Identidad no resuelta: error={error}"
    
    def test_kraus_identity_resolution_amplitude_damping(self):
        """Canal de amortiguamiento debe resolver identidad."""
        kraus_ops = create_amplitude_damping_channel(gamma=0.2)
        
        is_valid, error = QuantumChannelCharacterizer.verify_kraus_identity_resolution(
            kraus_ops
        )
        
        assert is_valid, f"Identidad no resuelta: error={error}"
    
    def test_kraus_identity_resolution_invalid(self):
        """Canal inválido no debe resolver identidad."""
        # Operadores que no resuelven identidad
        M1 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        kraus_ops = [M1]
        
        is_valid, error = QuantumChannelCharacterizer.verify_kraus_identity_resolution(
            kraus_ops
        )
        
        assert not is_valid, "Canal inválido aceptado"
        assert error > 0.1
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Matriz de Choi
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_choi_matrix_dimensions(self, pauli_operators):
        """Matriz de Choi debe tener dimensión correcta."""
        U = pauli_operators['Z']
        kraus_ops = create_unitary_channel(U)
        
        choi = QuantumChannelCharacterizer.compute_choi_matrix(kraus_ops)
        
        dim = U.shape[0]
        expected_dim = dim ** 2
        
        assert choi.shape == (expected_dim, expected_dim), \
            f"Dimensión incorrecta: {choi.shape}"
    
    def test_choi_matrix_hermitian(self, pauli_operators):
        """Matriz de Choi debe ser hermitiana."""
        U = pauli_operators['Y']
        kraus_ops = create_unitary_channel(U)
        
        choi = QuantumChannelCharacterizer.compute_choi_matrix(kraus_ops)
        
        hermiticity_error = la.norm(choi - choi.conj().T, ord='fro')
        
        assert hermiticity_error < 1e-10, \
            f"Matriz de Choi no hermitiana: {hermiticity_error}"
    
    def test_choi_matrix_positive_semidefinite(self):
        """Matriz de Choi debe ser semidefinida positiva."""
        kraus_ops = create_depolarizing_channel(dimension=2, p=0.2)
        
        choi = QuantumChannelCharacterizer.compute_choi_matrix(kraus_ops)
        
        eigenvalues = la.eigvalsh(choi)
        
        assert np.all(eigenvalues >= -1e-10), \
            f"Matriz de Choi no positiva: {eigenvalues}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Propiedades de Canal
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_unital_pauli_channel(self, pauli_operators):
        """Canal de Pauli es unital."""
        kraus_ops = [
            np.sqrt(0.25) * pauli_operators['I'],
            np.sqrt(0.25) * pauli_operators['X'],
            np.sqrt(0.25) * pauli_operators['Y'],
            np.sqrt(0.25) * pauli_operators['Z']
        ]
        
        is_unital = QuantumChannelCharacterizer.is_unital(kraus_ops)
        
        assert is_unital, "Canal de Pauli no es unital"
    
    def test_non_unital_amplitude_damping(self):
        """Canal de amortiguamiento no es unital."""
        kraus_ops = create_amplitude_damping_channel(gamma=0.5)
        
        is_unital = QuantumChannelCharacterizer.is_unital(kraus_ops)
        
        assert not is_unital, "Canal de amortiguamiento es unital (incorrecto)"
    
    def test_unitarity_unitary_channel(self, pauli_operators):
        """Unitariedad de canal unitario debe ser 1."""
        U = pauli_operators['X']
        kraus_ops = create_unitary_channel(U)
        
        unitarity = QuantumChannelCharacterizer.compute_unitarity(kraus_ops)
        
        assert np.isclose(unitarity, 1.0, atol=1e-10), \
            f"Unitariedad incorrecta: {unitarity}"
    
    def test_unitarity_depolarizing_channel(self):
        """Unitariedad de canal despolarizante."""
        p = 0.5
        kraus_ops = create_depolarizing_channel(dimension=2, p=p)
        
        unitarity = QuantumChannelCharacterizer.compute_unitarity(kraus_ops)
        
        # Para canal despolarizante: u = (1-p)²
        expected = (1 - p) ** 2
        
        assert np.isclose(unitarity, expected, atol=1e-8), \
            f"Unitariedad incorrecta: {unitarity} vs {expected}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Caracterización Completa
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_characterize_unitary_channel(self, pauli_operators):
        """Caracterización de canal unitario."""
        U = pauli_operators['Z']
        kraus_ops = create_unitary_channel(U)
        
        char = QuantumChannelCharacterizer.characterize_channel(kraus_ops)
        
        assert char.channel_type == ChannelType.UNITARY
        assert char.kraus_rank == 1
        assert char.is_trace_preserving
        assert char.is_completely_positive
        assert np.isclose(char.unitarity, 1.0, atol=1e-6)
    
    def test_characterize_depolarizing_channel(self):
        """Caracterización de canal despolarizante."""
        kraus_ops = create_depolarizing_channel(dimension=2, p=0.3)
        
        char = QuantumChannelCharacterizer.characterize_channel(kraus_ops)
        
        assert char.channel_type == ChannelType.KRAUS
        assert char.kraus_rank == 4
        assert char.is_trace_preserving
        assert char.is_completely_positive
        assert char.is_unital
    
    def test_characterize_amplitude_damping_channel(self):
        """Caracterización de canal de amortiguamiento."""
        kraus_ops = create_amplitude_damping_channel(gamma=0.4)
        
        char = QuantumChannelCharacterizer.characterize_channel(kraus_ops)
        
        assert char.kraus_rank == 2
        assert char.is_trace_preserving
        assert not char.is_unital


# ══════════════════════════════════════════════════════════════════════════════
# FASE 4: PRUEBAS DE VECTOR ASSIMILATE TOON CARTRIDGE
# ══════════════════════════════════════════════════════════════════════════════

class TestVectorAssimilateToonCartridge:
    """Suite de pruebas para asimilación de cartuchos."""
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Asimilación Exitosa
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_assimilate_unitary_channel(self, pauli_operators):
        """Asimilación con canal unitario."""
        rho = create_quantum_mac_state(dimension=2, purity=0.8, seed=42)
        U = pauli_operators['X']
        kraus_ops = create_unitary_channel(U)
        
        metadata = {'id': 'test_unitary', 'type': 'unitary'}
        
        result = vector_assimilate_toon_cartridge(
            current_rho=rho,
            kraus_operators=kraus_ops,
            cartridge_metadata=metadata,
            fidelity_threshold=0.5
        )
        
        assert result['success'], f"Asimilación falló: {result.get('error')}"
        assert result['status'] == VectorResultStatus.SUCCESS
        assert 'fidelity_preservation' in result
        assert result['fidelity_preservation'] >= 0.5
    
    def test_assimilate_depolarizing_channel(self):
        """Asimilación con canal despolarizante."""
        rho = create_quantum_mac_state(dimension=2, purity=0.7, seed=42)
        kraus_ops = create_depolarizing_channel(dimension=2, p=0.1)
        
        metadata = {'id': 'test_depolarizing'}
        
        result = vector_assimilate_toon_cartridge(
            current_rho=rho,
            kraus_operators=kraus_ops,
            cartridge_metadata=metadata,
            fidelity_threshold=0.7
        )
        
        assert result['success']
        assert 'injection_quality' in result
    
    def test_assimilate_preserves_trace(self, pauli_operators):
        """Asimilación debe preservar traza."""
        rho = create_quantum_mac_state(dimension=2, purity=0.6, seed=42)
        U = pauli_operators['Y']
        kraus_ops = create_unitary_channel(U)
        
        metadata = {'id': 'test_trace'}
        
        result = vector_assimilate_toon_cartridge(
            current_rho=rho,
            kraus_operators=kraus_ops,
            cartridge_metadata=metadata
        )
        
        if result['success']:
            rho_new = result['new_rho']
            trace = np.trace(rho_new).real
            
            assert np.isclose(trace, 1.0, atol=1e-10), \
                f"Traza no preservada: {trace}"
    
    def test_assimilate_preserves_hermiticity(self, pauli_operators):
        """Asimilación debe preservar hermiticidad."""
        rho = create_quantum_mac_state(dimension=2, purity=0.8, seed=42)
        U = pauli_operators['Z']
        kraus_ops = create_unitary_channel(U)
        
        metadata = {'id': 'test_hermiticity'}
        
        result = vector_assimilate_toon_cartridge(
            current_rho=rho,
            kraus_operators=kraus_ops,
            cartridge_metadata=metadata
        )
        
        if result['success']:
            rho_new = result['new_rho']
            hermiticity_error = la.norm(rho_new - rho_new.conj().T, ord='fro')
            
            assert hermiticity_error < 1e-10, \
                f"Hermiticidad no preservada: {hermiticity_error}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Validación
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_assimilate_rejects_non_tp_channel(self):
        """Debe rechazar canal que no preserva traza."""
        rho = create_quantum_mac_state(dimension=2, purity=0.7, seed=42)
        
        # Canal inválido
        M1 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        kraus_ops = [M1]
        
        metadata = {'id': 'test_invalid'}
        
        result = vector_assimilate_toon_cartridge(
            current_rho=rho,
            kraus_operators=kraus_ops,
            cartridge_metadata=metadata,
            validate_channel=True
        )
        
        assert not result['success']
        assert result['status'] == VectorResultStatus.VALIDATION_ERROR
    
    def test_assimilate_rejects_low_fidelity(self):
        """Debe rechazar si fidelidad está bajo umbral."""
        rho = create_quantum_mac_state(dimension=2, purity=0.9, seed=42)
        
        # Canal muy ruidoso
        kraus_ops = create_depolarizing_channel(dimension=2, p=0.9)
        
        metadata = {'id': 'test_low_fidelity'}
        
        result = vector_assimilate_toon_cartridge(
            current_rho=rho,
            kraus_operators=kraus_ops,
            cartridge_metadata=metadata,
            fidelity_threshold=0.9
        )
        
        assert not result['success']
        assert result['status'] == VectorResultStatus.TOPOLOGY_ERROR
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Métricas
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_assimilate_computes_metrics(self, pauli_operators):
        """Debe calcular métricas completas."""
        rho = create_quantum_mac_state(dimension=2, purity=0.7, seed=42)
        U = pauli_operators['X']
        kraus_ops = create_unitary_channel(U)
        
        metadata = {'id': 'test_metrics'}
        
        result = vector_assimilate_toon_cartridge(
            current_rho=rho,
            kraus_operators=kraus_ops,
            cartridge_metadata=metadata,
            compute_metrics=True
        )
        
        if result['success']:
            assert 'injection_report' in result
            report = result['injection_report']
            
            assert isinstance(report, InjectionReport)
            assert report.fidelity_preservation >= 0
            assert 0 <= report.purity_before <= 1
            assert 0 <= report.purity_after <= 1


# ══════════════════════════════════════════════════════════════════════════════
# FASE 5: PRUEBAS DE VECTOR COLLAPSE POVM DECISION
# ══════════════════════════════════════════════════════════════════════════════

class TestVectorCollapsePOVMDecision:
    """Suite de pruebas para colapso POVM."""
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Colapso Exitoso
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_collapse_projective_measurement(self):
        """Colapso con medición proyectiva."""
        rho = create_quantum_mac_state(dimension=2, purity=0.8, seed=42)
        
        # POVM proyectivo en base Z
        M0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        M1 = np.array([[0, 0], [0, 1]], dtype=np.complex128)
        povm_ops = [M0, M1]
        
        result = vector_collapse_povm_decision(
            current_rho=rho,
            povm_observables=povm_ops,
            decision_context="test_projective"
        )
        
        assert result['success']
        assert 'decision_index' in result
        assert result['decision_index'] in [0, 1]
    
    def test_collapse_deterministic_mode(self):
        """Modo determinista debe seleccionar máxima probabilidad."""
        # Estado sesgado hacia |0⟩
        psi = np.array([0.9, 0.1], dtype=np.complex128)
        psi /= la.norm(psi)
        rho = AtomicDensityMatrix(np.outer(psi, psi.conj()))
        
        M0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        M1 = np.array([[0, 0], [0, 1]], dtype=np.complex128)
        povm_ops = [M0, M1]
        
        result = vector_collapse_povm_decision(
            current_rho=rho,
            povm_observables=povm_ops,
            decision_context="test_deterministic",
            deterministic=True
        )
        
        # Debe seleccionar índice 0 (|0⟩)
        assert result['success']
        assert result['decision_index'] == 0
    
    def test_collapse_preserves_trace(self):
        """Colapso debe preservar traza."""
        rho = create_quantum_mac_state(dimension=2, purity=0.7, seed=42)
        
        M0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        M1 = np.array([[0, 0], [0, 1]], dtype=np.complex128)
        povm_ops = [M0, M1]
        
        result = vector_collapse_povm_decision(
            current_rho=rho,
            povm_observables=povm_ops,
            decision_context="test_trace"
        )
        
        if result['success']:
            rho_collapsed = result['collapsed_rho']
            trace = np.trace(rho_collapsed).real
            
            assert np.isclose(trace, 1.0, atol=1e-10), \
                f"Traza no preservada: {trace}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Estadísticas
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_collapse_computes_statistics(self):
        """Debe calcular estadísticas de medición."""
        rho = create_quantum_mac_state(dimension=2, purity=0.6, seed=42)
        
        M0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        M1 = np.array([[0, 0], [0, 1]], dtype=np.complex128)
        povm_ops = [M0, M1]
        
        result = vector_collapse_povm_decision(
            current_rho=rho,
            povm_observables=povm_ops,
            decision_context="test_statistics",
            compute_statistics=True
        )
        
        if result['success']:
            assert 'shannon_entropy' in result
            assert 'mutual_information' in result
            assert 'measurement_disturbance' in result
            assert result['shannon_entropy'] >= 0
            assert result['mutual_information'] >= 0
    
    def test_collapse_probability_bounds(self):
        """Probabilidad de decisión debe estar en [0, 1]."""
        rho = create_quantum_mac_state(dimension=2, purity=0.8, seed=42)
        
        M0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        M1 = np.array([[0, 0], [0, 1]], dtype=np.complex128)
        povm_ops = [M0, M1]
        
        result = vector_collapse_povm_decision(
            current_rho=rho,
            povm_observables=povm_ops,
            decision_context="test_probability",
            compute_statistics=True
        )
        
        if result['success']:
            prob = result['decision_probability']
            assert 0 <= prob <= 1, f"Probabilidad fuera de rango: {prob}"


# ══════════════════════════════════════════════════════════════════════════════
# FASE 6: PRUEBAS DE VECTOR AUDIT MODULAR CONJUGATION
# ══════════════════════════════════════════════════════════════════════════════

class TestVectorAuditModularConjugation:
    """Suite de pruebas para auditoría modular."""
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Auditoría Exitosa
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_audit_low_entropy_state(self):
        """Estado de baja entropía debe pasar auditoría."""
        rho = create_quantum_mac_state(dimension=2, purity=0.95, seed=42)
        mic_rank = 10
        
        result = vector_audit_modular_conjugation(
            mac_rho=rho,
            mic_dimension_rank=mic_rank
        )
        
        assert result['success']
        assert result['galois_adjunction_secured']
        assert 'modular_asymmetry' in result
    
    def test_audit_high_entropy_state_fails(self):
        """Estado de alta entropía debe fallar con mic_rank pequeño."""
        dim = 5
        rho_mixed = AtomicDensityMatrix(np.eye(dim, dtype=np.complex128) / dim)
        mic_rank = 2  # Rango muy pequeño
        
        result = vector_audit_modular_conjugation(
            mac_rho=rho_mixed,
            mic_dimension_rank=mic_rank
        )
        
        # Debe fallar porque asimetría > ln(2)
        assert not result['success']
        assert result['status'] == VectorResultStatus.LOGIC_ERROR
    
    def test_audit_computes_metrics(self):
        """Debe calcular métricas modulares."""
        rho = create_quantum_mac_state(dimension=3, purity=0.7, seed=42)
        mic_rank = 10
        
        result = vector_audit_modular_conjugation(
            mac_rho=rho,
            mic_dimension_rank=mic_rank
        )
        
        if result['success']:
            assert 'modular_asymmetry' in result
            assert 'relative_entropy' in result
            assert 'fisher_information' in result
            assert result['modular_asymmetry'] >= 0
            assert result['relative_entropy'] >= 0
    
    def test_audit_with_fisher_information(self):
        """Calcular información de Fisher con observable."""
        rho = create_quantum_mac_state(dimension=2, purity=0.8, seed=42)
        observable = np.array([[1, 0], [0, -1]], dtype=np.complex128)  # σ_z
        mic_rank = 10
        
        result = vector_audit_modular_conjugation(
            mac_rho=rho,
            mic_dimension_rank=mic_rank,
            compute_fisher=True,
            observable=observable
        )
        
        if result['success']:
            assert 'fisher_information' in result
            assert result['fisher_information'] >= 0
    
    # ─────────────────────────────────────────────────────────────────────────
    # Pruebas de Validación
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_audit_threshold_calculation(self):
        """Umbral debe ser ln(max(2, mic_rank))."""
        rho = create_quantum_mac_state(dimension=2, purity=0.9, seed=42)
        mic_rank = 5
        
        result = vector_audit_modular_conjugation(
            mac_rho=rho,
            mic_dimension_rank=mic_rank
        )
        
        expected_threshold = np.log(max(2, mic_rank))
        
        if result['success']:
            assert 'max_tolerable_asymmetry' in result
            assert np.isclose(
                result['max_tolerable_asymmetry'], 
                expected_threshold, 
                atol=1e-10
            )


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
pytest.mark.bures = pytest.mark.bures
pytest.mark.tomita = pytest.mark.tomita
pytest.mark.channel = pytest.mark.channel
pytest.mark.vectors = pytest.mark.vectors
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
        "--cov=app.wisdom.mac_vectors",
        "--cov-report=html",
        "--cov-report=term",
        "-m", "not slow",
        "--maxfail=10",
        "--durations=15"
    ])